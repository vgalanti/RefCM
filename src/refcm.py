import os
import ot
import json
import scipy
import scipy.sparse

import pulp as pl
import numpy as np
import scanpy as sc

from tqdm import tqdm
from typing import Union, List, Callable, Tuple, Dict, TypedDict, TypeAlias
from anndata import AnnData
from matchings import Matching
from embeddings import Embedder, HVGEmbedder


# config and logging setup
import config
import logging
import warnings

log = logging.getLogger(__name__)
# ignore pandas FutureWarnings originating from Scanpy's HVG method
warnings.simplefilter(action="ignore", category=FutureWarning)

# constants
MAX_CLUSTER_SIZE = 6000
MAX_HVG_SIZE = 30000


# Caching of compute-expensive WS costs
class CachedCost(TypedDict):
    # query and reference names
    q_id: str
    ref_id: str

    # parameters used in computation of WS cost
    target_sum: int | None
    n_top_genes: int
    num_iter_max: int
    max_hvg_size: int | None
    max_cluster_size: int | None

    # the computed ws costs
    costs: List[List[float]]


# TODO check if this requires log-normalized data
def dflt_clustering_func(adata: AnnData) -> AnnData:
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)  # , key_added="refcm_clusters"
    return adata


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
        self._embedder: Embedder = embedder
        self._cache_fpath: str = cache_fpath
        self._cache_load: bool = cache_load
        self._cache_save: bool = cache_save
        self._cache: List[CachedCost] = self._load_cache()

    def annotate(
        self,
        q: AnnData,
        q_id: str,
        ref: AnnData | List[AnnData],
        ref_id: str | List[str],
        ref_key_labels: str | List[str],
        q_key_clusters: str | None = None,
        q_clustering_func: str | Callable[[AnnData], AnnData] = "leiden",
    ) -> Matching:
        """
        Annotates a query dataset utilizing reference dataset(s)

        Parameters
        ----------
        q: AnnData
            Query data to annotate
        q_id: str
            Query id (name) for saving.
        ref: AnnData | List[AnnData]
            Reference datasets to utilize for annotation
        ref_id: str | List[str]
            Reference ids (names) for saving.
        ref_key_labels: str | List[str]
            'obs' keys in the reference datasets corresponding to their cell type labels
        q_key_clusters: str | None = None
            'obs' key corresponding to the query's current clustering
        q_clustering_func: str | Callable[[AnnData], AnnData] = "leiden"
            In case the query's clusters have not yet been computed, the clustering
            algorithm to apply.
            Resulting clustering should result in adata.obs[q_key_clusters]

        Returns
        -------
        Matching
            The query dataset along with its new annotations under 'obs'.
        """
        # annotate the dataset, creating new "obs" fields where necessary

        # if clusters are not pre-computed, compute them given the input function
        if q_key_clusters is None:
            if type(q_clustering_func) == str:
                if q_clustering_func != "leiden":
                    log.warning("Clustering other than default not yet supported")
                    log.warning("Defaulting to Leiden clustering")
                else:
                    log.info("Using default Leiden clustering")

                q_clustering_func = dflt_clustering_func
                q_key_clusters = "leiden"

            else:
                log.info("Using input clustering function")

            log.info("Clustering the query dataset")
            q_clustering_func(q)

        # check that the given key to the clusterings exists
        if q_key_clusters not in q.obs.columns:
            log.error("Key mapping to clusters invalid. Exiting.")
            return

        # add temporary metadata to the references for downstream tasks
        if not isinstance(ref, list):
            ref = [ref]
            ref_key_labels = [ref_key_labels]
            ref_id = [ref_id]

        for i, r in enumerate(ref):

            if ref_key_labels[i] not in r.obs.columns:
                log.error(f"Reference {ref_id[i]}'s .obs cluster key invalid.")
                return

            ltk, ktl = {}, {}
            r.uns["_ck"] = ref_key_labels[i]  # cluster key
            clusters = np.sort(np.unique(r.obs[r.uns["_ck"]]))
            for j, l in enumerate(clusters):
                ltk[l], ktl[j] = j, l

            r.uns["_ltk"] = ltk  # labels-to-clusters
            r.uns["_ktl"] = ktl  # clusters-to-labels
            r.uns["_nc"] = len(clusters)  # number of clusters
            r.uns["_id"] = ref_id[i]  # name id

        # create new refcm cluster and annotation observations for query anndata
        q.obs["refcm_clusters"] = -1
        q.obs["refcm_annot"] = ""

        clusters = np.sort(np.unique(q.obs[q_key_clusters]))
        for j, l in enumerate(clusters):
            q.obs.loc[q.obs[q_key_clusters] == l, "refcm_clusters"] = j

        q.uns["_nc"] = len(clusters)
        q.uns["_id"] = q_id

        # proceed to mapping query to reference
        m, c, ref_ktl = self._match(q, ref)

        # annotate based on the matching information.
        # TODO handle the ambiguous case where query is mapped to multiple reference clusters
        for i in range(len(clusters)):
            mapped = np.argmax(m[i] == 1)  # find first occurence of a mapping
            if m[i][mapped] == 0:  # meaning that it does not get mapped anywhere
                label = None
            else:
                label = ref_ktl[mapped]
            q.obs.loc[q.obs["refcm_clusters"] == i, "refcm_annot"] = label

        # save matching information under 'uns' (i.e. unstructured data)
        q.uns["refcm_mapping"] = m
        q.uns["refcm_costs"] = c
        q.uns["refcm_ref_ktl"] = ref_ktl

        # cleanup changes to reference datasets 'uns' during the way
        for r in ref:
            r.uns.pop("_ck", None)
            r.uns.pop("_ltk", None)
            r.uns.pop("_ktl", None)
            r.uns.pop("_nc", None)
            r.uns.pop("_id", None)

        return Matching(q, ref, q_id, ref_id)

    def _match(
        self,
        q: AnnData,
        ref: List[AnnData],
    ) -> Tuple[np.ndarray, np.ndarray, dict[int, str]]:
        """
        Matches a query Dataset to a Reference dataset(s).

        Parameters
        ----------
        q: AnnData
            query data to map
        ref: List[AnnData]
            reference dataset(s)

        Returns
        -------
        np.ndarray
            the final query -> reference mapping matrix
            Shape: (n_query_clusters, n_reference_clusters)
        np.ndarray
            the final query -> reference mapping cost matrix
            Shape: (n_query_clusters, n_reference_clusters)
        dict[int, str]
            the merged reference key-to-label correspondence

        NOTE
        ----
        * discovery_threshold assumes that all costs are non-positive.
        """

        costs = [self._matching_cost(q, r) for r in ref]
        cost, ref_ktl = self._merge_costs(q, ref, costs)

        c = cost.copy()
        if self._discovery_threshold is not None:
            c += abs(cost.min() * self._discovery_threshold)

        m = self.lp_match(c)
        return m, cost, ref_ktl

    def _merge_costs(
        self,
        q: AnnData,
        ref: List[AnnData],
        costs: List[np.ndarray],
    ) -> Tuple[np.ndarray, dict[int, str]]:
        """
        Merges the matching costs of query to multi-reference tasks and
        establishes a merged reference key-to-label correspondence.

        Parameters
        ----------
        q: AnnData
            query dataset
        ref: [AnnData]
            reference dataset
        costs: List[np.ndarray]
            list of matching costs between the query set and the references
            Shape: [ (n_query_clusters, n_reference_clusters) ]

        Returns
        -------
        np.ndarray
            the merged query -> reference cost matrix
        dict[int, str]
            the merged reference key-to-label correspondence
        """
        # get all available cell types across all reference datasets
        ref_labels = sum([[*r.uns["_ktl"].values()] for r in ref], [])
        ref_labels = np.sort(np.unique(ref_labels))

        q_n_labels = q.uns["_nc"]
        cost = np.zeros((q_n_labels, len(ref_labels)))
        for qc in range(q_n_labels):
            for rc, rl in enumerate(ref_labels):
                cost[qc][rc] = np.average(
                    [
                        costs[i][qc][r.uns["_ltk"].get(rl)]
                        for i, r in enumerate(ref)
                        if r.uns["_ltk"].get(rl) is not None
                    ]
                )

        ref_ktl = {k: v for k, v in enumerate(ref_labels)}
        return cost, ref_ktl

    def _matching_cost(self, q: AnnData, ref: AnnData) -> np.ndarray:
        """
        Computes the matching cost between pairs of query and reference clusters

        Parameters
        ----------
        q: AnnData
            query dataset
        ref: AnnData
            reference dataset

        Returns
        -------
        np.ndarray
            the query -> reference wasserstein distances
            Shape: (n_query_clusters, n_reference_clusters)
        """
        # TODO build upon just WS with global geometry and
        # prior species/measurement location/measurement tech
        # information, either here or in the "merge costs" section

        # check whether costs have already been saved
        q_id, ref_id = q.uns["_id"], ref.uns["_id"]

        if self._cache_load:
            c = self._cache_get(q_id, ref_id)
            if c is not None:
                log.debug(f"Using costs for {q_id}->{ref_id} found in cache.")
                return np.array(c)

        # otherwise, compute them
        c = self._ws(q, ref)

        if self._cache_save:
            self._update_cache(q_id, ref_id, c.tolist())

        return c

    def _ws(self, q: AnnData, ref: AnnData) -> np.ndarray:
        """
        Computes the Wasserstein distance between pairs of query and reference clusters

        Parameters
        ----------
        q: Dataset
            query dataset
        ref: Dataset
            reference dataset

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
        self._embedder.fit(q, ref)

        # compute wasserstein distances between cluster pairs
        log.debug("Computing Wasserstein distances.")
        ws = np.zeros((q.uns["_nc"], ref.uns["_nc"]))

        tqdm_bar = "|{bar:16}| [{percentage:>6.2f}% ] : {elapsed}"
        with tqdm(total=q.uns["_nc"] * ref.uns["_nc"], bar_format=tqdm_bar) as pbar:
            for qc in range(q.uns["_nc"]):
                x_qc = q[q.obs["refcm_clusters"] == qc]
                x_qc = self._embedder.embed(x_qc)

                for rc in range(ref.uns["_nc"]):
                    x_rc = ref[ref.obs[ref.uns["_ck"]] == ref.uns["_ktl"][rc]]
                    x_rc = self._embedder.embed(x_rc)

                    a, b = unif(len(x_qc)), unif(len(x_rc))
                    M = -x_qc @ x_rc.T

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

    def _cache_get(self, q_id: str, ref_id: str) -> List[List[float]] | None:
        """
        Retrieves mapping costs from cache, if available.

        Parameters
        ----------
        q_id: str
            Query id (name).
        ref_id: str
            Reference id (name).

        Returns
        -------
        List[List[float]] | None:
            The corresponding mapping cost, if it exists.
        """

        pred = lambda c: (
            c["q_id"] == q_id
            and c["ref_id"] == ref_id
            and c["num_iter_max"] == self._num_iter_max
            and c["target_sum"] == self._target_sum
            and c["n_top_genes"] == self._n_top_genes
            and c["max_hvg_size"] == self._max_hvg_size
            and c["max_cluster_size"] == self._max_cluster_size
        )
        result = list(filter(pred, self._cache))

        if len(result) > 1:
            log.error(f"Corrupted Cache: conflicting entries for {q_id}->{ref_id}")
            raise Exception(f"Corrupted Cache {self._cache_fpath}.")

        return result[0]["costs"] if len(result) == 1 else None

    def _update_cache(self, q_id: str, ref_id: str, mcosts: List[List[float]]) -> None:
        """
        Updates the mapping costs cache file.

        Parameters
        ----------
        q_id: str
            Query id (name) for saving.
        ref_id: str
            Reference id (name) for saving.
        mcosts: List[List[float]]
            Pair mapping costs to add to the cache.
        """

        if self._cache_get(q_id, ref_id) is not None:
            return

        log.debug(f"Saving {q_id}->{ref_id} mapping costs to {self._cache_fpath}.")
        self._cache.append(
            {
                "q_id": q_id,
                "ref_id": ref_id,
                "target_sum": self._target_sum,
                "n_top_genes": self._n_top_genes,
                "num_iter_max": self._num_iter_max,
                "max_hvg_size": self._max_hvg_size,
                "max_cluster_size": self._max_cluster_size,
                "costs": mcosts,
            }
        )

        with open(self._cache_fpath, "w") as f:
            json.dump(self._cache, f, indent=4, ensure_ascii=False)
