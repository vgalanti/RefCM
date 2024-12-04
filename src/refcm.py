import os
import ot
import json

import pulp as pl
import numpy as np

from tqdm import tqdm
from typing import List, Tuple, TypedDict
from anndata import AnnData
from matchings import Matching
from embeddings import Embedder, HVGEmbedder
from sklearn.metrics.pairwise import pairwise_distances


# config and logging setup
import config
import logging
import warnings

log = logging.getLogger(__name__)
# ignore pandas FutureWarnings originating from Scanpy's HVG method
warnings.simplefilter(action="ignore", category=FutureWarning)


# Caching of compute-expensive WS costs
class CachedCost(TypedDict):
    # relevant uids
    q_uid: str
    ref_uid: str
    embed_uid: str

    # additional WS compute param
    pdist: str
    num_iter_max: int

    # the computed ws costs
    costs: List[List[float]]


class RefCM:
    def __init__(
        self,
        max_merges: int = 1,
        max_splits: int = 1,
        n_target_clusters: int | None = None,
        discovery_threshold: float | None = 0.2,
        verbose_solver: bool = False,
        num_iter_max: int = 1e7,
        embedder: Embedder = HVGEmbedder(),
        pdist: str = "default",
        cache_load: bool = True,
        cache_save: bool = True,
        cache_fpath: str = config.CACHE,
    ) -> None:
        """
        Initializes RefCM instance

        Parameters
        ----------
        max_merges: int = 1
            maximum number of query clusters that can merge to a single
            reference cluster, -1 indicates infinite
        max_splits:int = 1
            maximum number of reference clusters that a single query cluster
            can map to, -1 indicates infinite
        n_target_cluster: int | None = None
            maximum number of target clusters to map to
        discovery_threshold: float | None = 0.2
            cost threshold (as % of lowest cost) over which a cluster does
            not get mapped and is instead considered a potentially new cell type
        verbose_solver: bool = False
            whether PuLP consol output should be silenced
        num_iter_max: int = 1e7
            max number of iterations for the emd optimization
        embedder: Embedder = HVGEmbedder()
            the embedder to use.
        pdist: str = 'default'
            the pairwise distance to use between query and reference cells.
            str compatible with sklearn's pairwise_distances.
            defaults to -dot(a, b)
        cache_load: bool = True
            Whether to load existing mapping costs (if available) from cache.
        cache_save: bool = True,
            Whether to save new mapping costs to cache.
        cache_fpath: str = config.CACHE
            Path to stored mapping costs.

        """
        log.info("NOTE: raw counts expected in anndata .X attributes.")

        self._max_merges: int = max_merges
        self._max_splits: int = max_splits
        self._n_target_clusters: int | None = n_target_clusters
        self._discovery_threshold: float | None = discovery_threshold
        self._verbose_solver: bool = verbose_solver
        self._num_iter_max: int = num_iter_max
        self._pdist: str = pdist
        self._embedder: Embedder = embedder
        self._cache_fpath: str = cache_fpath
        self._cache_load: bool = cache_load
        self._cache_save: bool = cache_save
        self._cache: List[CachedCost] = self._load_cache()

    def setref(self, ref: AnnData, ref_id: AnnData, ref_key: str) -> None:
        """
        Sets the reference dataset to be used during annotation.

        Parameters
        ----------
        ref: AnnData
            Reference dataset to to use.
        ref_id: str
            Reference id (name) for saving.
        ref_key: str
            'obs' key in the reference dataset where ground truth labels lie.
        """
        if ref_key not in ref.obs.columns:
            log.error(f"Reference .obs does not contain {ref_key}. Exiting.")
            return

        self.ref = ref
        self.ref_id = ref_id
        self.ref_key = ref_key

        # label-to-key and and key-to-label maps
        self.ref_ltk, self.ref_ktl = {}, {}

        labels = np.sort(np.unique(self.ref.obs[self.ref_key]))
        for j, l in enumerate(labels):
            self.ref_ltk[l], self.ref_ktl[j] = j, l

        self.ref_nc = len(labels)
        self.ref_labels = labels

    def annotate(
        self,
        q: AnnData,
        q_id: str,
        q_key: str,
    ) -> Matching:
        """
        Annotates a query dataset.

        Parameters
        ----------
        q: AnnData
            Query data to annotate/
        q_id: str
            Query id (name) for saving.
        q_key: str
            'obs' key containing query's current clustering

        Returns
        -------
        Matching
            The query dataset along with its new annotations under 'obs'.
        """

        if not self.ref:
            log.error("Set reference before annotating. Exiting.")
            return

        # check that the given key to the clusterings exists
        if q_key not in q.obs.columns:
            log.error(f"Query .obs does not contain {q_key}. Exiting.")
            return

        self.q = q
        self.q_id = q_id
        self.q_key = q_key

        # create new refcm cluster and annotation observations for query anndata
        self.q.obs["refcm_clusters"] = -1
        self.q.obs["refcm_annot"] = ""

        clusters = np.sort(np.unique(self.q.obs[self.q_key]))
        for j, l in enumerate(clusters):
            self.q.obs.loc[self.q.obs[self.q_key] == l, "refcm_clusters"] = j

        self.q_nc = len(clusters)

        # proceed to mapping query to reference
        m, c = self._match()

        # annotate based on the matching information.
        # TODO handle the ambiguous case where query is mapped to multiple reference clusters
        for i in range(self.q_nc):
            mapped = np.argmax(m[i] == 1)  # find first occurence of a mapping
            if m[i][mapped] == 0:  # meaning that it does not get mapped anywhere
                label = "novel"
            else:
                label = self.ref_ktl[mapped]
            self.q.obs.loc[self.q.obs["refcm_clusters"] == i, "refcm_annot"] = label

        # save matching information under 'uns' (i.e. unstructured data)
        # TODO if we want to keep this or not
        q.uns["refcm_mapping"] = m
        q.uns["refcm_costs"] = c
        q.uns["refcm_ref_ktl"] = self.ref_ktl

        return Matching(self.q, self.ref, self.q_id, self.ref_id)

    def _match(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """
        Matches a query Dataset to a reference dataset.

        Returns
        -------
        np.ndarray
            the final query -> reference mapping matrix
            Shape: (n_query_clusters, n_reference_clusters)
        np.ndarray
            the final query -> reference mapping cost matrix
            Shape: (n_query_clusters, n_reference_clusters)

        NOTE
        ----
        * discovery_threshold assumes that all costs are non-positive.
        """

        # compute matching costs
        if self._cache_load and (c := self._cache_get()) is not None:
            log.debug(f"Using costs for {self.q_id}->{self.ref_id} found in cache.")
            cost = np.array(c)

        else:
            cost = self._ws()
            if self._cache_save:
                self._update_cache(cost.tolist())

        # normalize between -1 and 0
        mx, mn = cost.max(), cost.min()
        cost = (cost - mx) / (mx - mn)

        c = cost.copy()
        if self._discovery_threshold is not None:
            c[c >= -self._discovery_threshold] = 1e2  # > 0

        # perform LP matching
        m = self.lp_match(c)
        return m, cost

    def _ws(self) -> np.ndarray:
        """
        Computes the Wasserstein distance between pairs of query and reference clusters

        Returns
        -------
        np.ndarray
            the query -> reference wasserstein distances
            Shape: (n_query_clusters, n_reference_clusters)

        TODO
        ----
        * test multiprocessing/threading option
        """
        unif = lambda s: np.ones(s) / s

        # fit the embedding to query and reference pair
        self._embedder.fit(self.q, self.ref)

        # compute wasserstein distances between cluster pairs
        log.debug("Computing Wasserstein distances.")
        ws = np.zeros((self.q_nc, self.ref_nc))

        tqdm_bar = "|{bar:16}| [{percentage:>6.2f}% ] : {elapsed}"
        with tqdm(total=self.q_nc * self.ref_nc, bar_format=tqdm_bar) as pbar:
            for qc in range(self.q_nc):
                x_qc = self.q[self.q.obs["refcm_clusters"] == qc]
                x_qc = self._embedder.embed(x_qc)

                for rc in range(self.ref_nc):
                    x_rc = self.ref[self.ref.obs[self.ref_key] == self.ref_ktl[rc]]
                    x_rc = self._embedder.embed(x_rc)

                    a, b = unif(len(x_qc)), unif(len(x_rc))

                    M = (
                        -x_qc @ x_rc.T
                        if self._pdist == "default"
                        else pairwise_distances(x_qc, x_rc, self._pdist)
                    )

                    ws[qc][rc] = ot.emd2(a, b, M, numItermax=self._num_iter_max)
                    pbar.update(1)

        return ws

    def lp_match(self, cost: np.ndarray) -> np.ndarray:
        """
        Matches a query Dataset to a Reference dataset using an IP matching
        formulation.

        Parameters
        ----------
        cost: np.ndarray
            query -> reference cluster mapping cost
            Shape: (n_query_clusters, n_reference_clusters)

        Returns
        -------
        np.ndarray
            the query -> reference matching, where the i'th entry represents
            the reference cluster(s) to which the i'th cluster in query maps to.
            Shape: (n_query_clusters, n_reference_clusters)
        """
        # create LP problem
        prob = pl.LpProblem()

        # Decision variables for each edge
        q_n_labels, ref_n_labels = cost.shape
        q_n, ref_n = np.arange(q_n_labels), np.arange(ref_n_labels)
        e = pl.LpVariable.dicts("e", (q_n, ref_n), 0, 1, pl.LpInteger)

        # Objective function to minimize
        prob += pl.lpSum([cost[u][v] * e[u][v] for u in q_n for v in ref_n])

        # Constraints for merging/splitting (defaults to bipartite matching)
        if self._max_splits >= 0:
            for u in q_n:
                prob += pl.lpSum([e[u][v] for v in ref_n]) <= self._max_splits

        if self._max_merges >= 0:
            for v in ref_n:
                prob += pl.lpSum([e[u][v] for u in q_n]) <= self._max_merges

        # Big-M constraints to indicate which of the target clusters
        # are used, so we can control how many clusters we map to
        if self._n_target_clusters is not None:
            w = pl.LpVariable.dicts("w", ref_n, 0, 1, pl.LpInteger)
            for i in ref_n:
                prob += w[i] <= pl.lpSum([e[v][i] for v in q_n])
                prob += pl.lpSum([e[v][i] for v in q_n]) <= w[i] * len(q_n)
            prob += pl.lpSum([w[i] for i in ref_n]) <= self._n_target_cluster
        # solve the problem and return it (msg=0 to suppress messages)
        log.debug(f"starting LP optimization")
        prob.solve(pl.GLPK_CMD(msg=self._verbose_solver))  # pl.PULP_CBC_CMD(msg=0))
        log.debug(f'optimization terminated w. status "{pl.LpStatus[prob.status]}"')

        # retrieve the matches
        m = np.zeros_like(cost)
        for v in prob.variables():
            if v.name[0] != "e":
                continue
            i, j = (int(x) for x in v.name.split("_")[1:])
            m[i, j] = v.varValue

        return m

    """############ utils ############"""

    def _load_cache(self) -> List[CachedCost]:
        """Loads cached mapping costs."""
        if not os.path.isfile(self._cache_fpath):
            log.debug(f"No existing cost-cache found ({self._cache_fpath}).")
            return []
        else:
            log.debug(f"Loading cached mapping costs from {self._cache_fpath}.")
            with open(self._cache_fpath, "r") as f:
                return json.load(f)

    def _cache_get(self) -> List[List[float]] | None:
        """
        Retrieves mapping costs from cache, if available.

        Returns
        -------
        List[List[float]] | None:
            The corresponding mapping cost, if it exists.
        """
        q_uid = f"{self.q_id}-{self.q_key}"
        ref_uid = f"{self.ref_id}-{self.ref_key}"

        # since WS costs are symmetric, we only save one entry
        # for (q, ref) and (ref, q); whichever has "lower" hash
        # alphabetically is q, and we transpose the costs if needed
        if flip := q_uid > ref_uid:
            temp = q_uid
            q_uid = ref_uid
            ref_uid = temp

        find = lambda c: (
            c["q_uid"] == q_uid
            and c["ref_uid"] == ref_uid
            and c["embed_uid"] == self._embedder.uid
            and c["pdist"] == self._pdist
            and c["num_iter_max"] == self._num_iter_max
        )
        result = list(filter(find, self._cache))

        if len(result) == 0:
            return None

        if len(result) > 1:
            log.error(f"Corrupted Cache: conflicting entries found.")
            raise Exception(f"Corrupted Cache {self._cache_fpath}.")

        r = result[0]
        return r["costs"] if not flip else np.array(r["costs"]).T.tolist()

    def _update_cache(self, mcosts: List[List[float]]) -> None:
        """
        Updates the mapping costs cache file.

        Parameters
        ----------
        q: AnnData
            Query for saving.
        ref: AnnData
            Reference for saving.
        mcosts: List[List[float]]
            Pair mapping costs to add to the cache.
        """
        q_uid = f"{self.q_id}-{self.q_key}"
        ref_uid = f"{self.ref_id}-{self.ref_key}"

        if flip := q_uid > ref_uid:
            temp = q_uid
            q_uid = ref_uid
            ref_uid = temp

        if self._cache_get() is not None:
            return

        log.debug(f"Saving mapping costs to {self._cache_fpath}.")
        self._cache.append(
            {
                "q_uid": q_uid,
                "ref_uid": ref_uid,
                "embed_uid": self._embedder.uid,
                "pdist": self._pdist,
                "num_iter_max": self._num_iter_max,
                "costs": mcosts if not flip else np.array(mcosts).T.tolist(),
            }
        )

        with open(self._cache_fpath, "w") as f:
            json.dump(self._cache, f, indent=4, ensure_ascii=False)
