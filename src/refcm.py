import os
import ot
import json
import scipy
import scipy.sparse

import pulp as pl
import numpy as np
import scanpy as sc

from tqdm import tqdm
from typing import Union, List, Callable, Tuple, Dict
from anndata import AnnData
from matchings import Matching


# config and logging setup
import config
import logging
import warnings

logging = logging.getLogger(__name__)
# ignore pandas FutureWarnings originating from Scanpy's HVG method
warnings.simplefilter(action="ignore", category=FutureWarning)

# constants
MAX_CLUSTER_SIZE = 6000
MAX_HVG_SIZE = 30000

# of the form [ query_id: {ref_id: [ mapping costs ]} ]
mcost_db = List[Dict[str, Dict[str, List[List[float]]]]]


# TODO check if this requires log-normalized data
def dflt_clustering_func(adata: AnnData) -> AnnData:
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)  # , key_added="refcm_clusters")
    return adata


class RefCM:
    def __init__(
        self,
        max_merges: int = 1,
        max_splits: int = 1,
        n_target_clusters: int | None = None,
        discovery_threshold: float | None = 0.2,
        verbose_solver: bool = False,
        numItermax: int = 2e5,
        n_top_genes: int = 1200,
        target_sum: int | None = None,
        subsample_large_clusters: bool = False,
        load_mcosts: bool = True,
        save_mcosts: bool = False,
        db_fpath: str = config.DB_FPATH,
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
        numItermax: int = 2e5
            max number of iterations for an emd optimization
        n_top_genes: int = 1200
            Number of highly variable genes (in self) to select
        target_sum: int | None = None
            Target row sum after normalization.
        subsample_large_clusters: bool = False
            Whether to subsample very large clusters to combat memory usage.
        load_mcosts: bool = True
            Whether to load existing mapping costs (if available).
        save_mcosts: bool = True,
            Whether to save new mapping costs.
        db_fpath: str = config.DB_FPATH
            Path to stored mapping costs.

        """
        logging.info("NOTE: raw counts expected in anndata .X attributes.")

        self._max_merges: int = max_merges
        self._max_splits: int = max_splits
        self._n_target_clusters: int | None = n_target_clusters
        self._discovery_threshold: float | None = discovery_threshold
        self._verbose_solver: bool = verbose_solver
        self._numItermax: int = numItermax
        self._n_top_genes: int = n_top_genes
        self._target_sum: int | None = target_sum
        self._subsample_large_clusters: bool = subsample_large_clusters
        self._db_fpath: str = db_fpath
        self._load_mcosts: bool = load_mcosts
        self._save_mcosts: bool = save_mcosts
        self._db: mcost_db = self._load_db()

    def annotate(
        self,
        q: AnnData,
        q_id: str,
        ref: Union[List[AnnData], AnnData],
        ref_id: Union[List[str], str],
        ref_key_labels: Union[List[str], str],
        q_key_clusters: str = None,
        q_clustering_func: Union[Callable[[AnnData], AnnData], str] = "leiden",
    ) -> Matching:
        """
        Annotates a query dataset utilizing reference dataset(s)

        Parameters
        ----------
        q: AnnData
            Query data to annotate
        q_id: str
            Query id (name) for saving.
        ref: Union[List[AnnData], AnnData]
            Reference datasets to utilize for annotation
        ref_id: Union[List[str], str]
            Reference ids (names) for saving.
        ref_key_labels: Union[List[str], str]
            'obs' keys in the reference datasets corresponding to their cell type labels
        q_key_clusters: str = None
            'obs' key corresponding to the query's current clustering
        q_clustering_func: Union[Callable[[AnnData], AnnData], str] = "leiden"
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
                    logging.warning("Clustering other than default not yet supported")
                    logging.warning("Defaulting to Leiden clustering")
                else:
                    logging.info("Using default Leiden clustering")

                q_clustering_func = dflt_clustering_func
                q_key_clusters = "leiden"

            else:
                logging.info("Using input clustering function")

            logging.info("Clustering the query dataset")
            q_clustering_func(q)

        # check that the given key to the clusterings exists
        if q_key_clusters not in q.obs.columns:
            logging.error("Key mapping to clusters invalid. Exiting.")
            return

        # add temporary metadata to the references for downstream tasks
        if not isinstance(ref, list):
            ref = [ref]
            ref_key_labels = [ref_key_labels]
            ref_id = [ref_id]

        for i, r in enumerate(ref):

            if ref_key_labels[i] not in r.obs.columns:
                logging.error(f"Reference {ref_id[i]}'s .obs cluster key invalid.")
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

        if self._load_mcosts:
            q_mcosts = self._db.get(q_id)
            c = q_mcosts.get(ref_id) if q_mcosts is not None else None
            if c is not None:
                logging.debug(f"Using costs for {q_id}->{ref_id} found in database.")
                return np.array(c)

        # otherwise, compute them
        c = self._ws(q, ref)

        if self._save_mcosts:
            self._update_db(q_id, ref_id, c.tolist())

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
        """
        unif = lambda s: np.ones(s) / s

        # log1p normalize query and referenc
        q_sums = q.X.sum(axis=1).reshape((-1, 1))
        ref_sums = ref.X.sum(axis=1).reshape((-1, 1))

        if self._target_sum is not None:
            q.X = self._target_sum * q.X / q_sums
            ref.X = self._target_sum * ref.X / ref_sums

        q.X = np.log1p(q.X)
        ref.X = np.log1p(ref.X)

        # filter out the most relevant genes
        gs = self.joint_gene_subset(q, ref)

        # compute wasserstein distances between cluster pairs
        logging.debug("Computing Wasserstein distances.")
        ws = np.zeros((q.uns["_nc"], ref.uns["_nc"]))

        tqdm_bar = "|{bar:16}| [{percentage:>6.2f}% ] : {elapsed}"
        with tqdm(total=q.uns["_nc"] * ref.uns["_nc"], bar_format=tqdm_bar) as pbar:
            for qc in range(q.uns["_nc"]):
                x_qc = q[q.obs["refcm_clusters"] == qc, gs].X
                x_qc = self._memreg(x_qc)

                for rc in range(ref.uns["_nc"]):
                    x_rc = ref[ref.obs[ref.uns["_ck"]] == ref.uns["_ktl"][rc], gs].X
                    x_rc = self._memreg(x_rc)

                    a, b = unif(len(x_qc)), unif(len(x_rc))
                    M = -x_qc @ x_rc.T

                    ws[qc][rc] = ot.emd2(a, b, M, numItermax=self._numItermax)
                    pbar.update(1)

        # convert query and reference back to raw counts
        q.X = np.expm1(q.X)
        ref.X = np.expm1(ref.X)

        if self._target_sum is not None:
            q.X *= q_sums / self._target_sum
            ref.X *= ref_sums / self._target_sum

        return ws

    def _memreg(self, x: np.ndarray) -> np.ndarray:
        """
        Regulates memory usage for large clusters.

        Parameters
        ----------
        x: np.ndarray | scipy sparse matrix
            The cluster to preprocess.

        Returns
        -------
        np.ndarray:
            The normalized/preprocessed cluster in non-sparse format.
        """

        if self._subsample_large_clusters and x.shape[0] > MAX_CLUSTER_SIZE:
            sbs = np.random.choice(x.shape[0], MAX_CLUSTER_SIZE, replace=False)
            x = x[sbs]

        if scipy.sparse.issparse(x):
            x = x.toarray()

        return x

    # TODO add the marker gene options here
    def joint_gene_subset(self, q: AnnData, ref: AnnData) -> List[str]:
        """
        Select a highly variable gene subset that is common to both query and reference.

        Parameters
        ----------
        q: Dataset
            query dataset
        ref: Dataset
            reference dataset

        Returns
        -------
        List[str]
            A subset of gene identifiers to use.
        """
        logging.debug("Selecting joint gene subset for query and reference datasets")

        # select highly variable genes from both query and reference datasets
        hvg = set(self._hvg(q)).union(self._hvg(ref))

        # intersect genes to make sure we don't have discrepencies
        common_genes = np.intersect1d(q.var_names, ref.var_names)

        # only note those that are both highly variable and common to the other
        gene_subset = list(set(hvg) & set(common_genes))
        logging.debug(f"Using {len(gene_subset)} genes.")

        return gene_subset

    def _hvg(self, ds: AnnData) -> List[str]:
        """
        Returns an AnnData's highly-variable genes list.

        Parameters
        ----------
        ds: AnnData
            The dataset to compute the hvg genes of.

        Returns
        -------
        List[str]:
            List of genes with the highest variability.
        """
        n = ds.X.shape[0]
        sbset = np.random.choice(n, min(n, MAX_HVG_SIZE), replace=False)
        tmp = ds[sbset].copy()

        sc.pp.highly_variable_genes(tmp, n_top_genes=self._n_top_genes)

        return tmp.var["highly_variable"][tmp.var["highly_variable"]].index.to_list()

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
        logging.debug(f"starting LP optimization")
        prob.solve(pl.GLPK_CMD(msg=self._verbose_solver))  # pl.PULP_CBC_CMD(msg=0))
        logging.debug(f'optimization terminated w. status "{pl.LpStatus[prob.status]}"')

        # retrieve the matches
        m = np.zeros_like(cost)
        for v in prob.variables():
            if v.name[0] != "e":
                continue
            i, j = (int(x) for x in v.name.split("_")[1:])
            m[i, j] = v.varValue

        return m

    """############ utils ############"""

    def _load_db(self) -> mcost_db:
        """Loads saved mapping costs."""
        if not os.path.isfile(self._db_fpath):
            logging.debug(f"No existing matching db cost file {self._db_fpath} found.")
            return {}
        else:
            logging.debug(f"Loading existing mapping costs from {self._db_fpath}.")
            with open(self._db_fpath, "r") as f:
                return json.load(f)

    def _update_db(self, q_id: str, ref_id: str, mcosts: List[List[float]]) -> None:
        """
        Updates the mapping costs database.

        Parameters
        ----------
        q_id: str
            Query id (name) for saving.
        ref_id: str
            Reference id (name) for saving.
        mcosts: List[List[float]]
            Pair mapping costs to add to the db.
        """
        logging.debug(f"Adding {q_id}->{ref_id} mapping costs to {self._db_fpath}.")

        if self._db.get(q_id) is None:
            self._db[q_id] = {ref_id: mcosts}
        else:
            self._db[q_id][ref_id] = mcosts

        with open(self._db_fpath, "w") as f:
            json.dump(self._db, f, sort_keys=True, indent=4, ensure_ascii=False)

    def existing_mcosts(
        self, with_q: Union[str, None] = None, with_ref: Union[str, None] = None
    ) -> List[str]:
        """
        Returns the names of saved/existing matching query -> reference mapping costs.

        Parameters
        ----------
        with_q: Union[str, None] = None
            Filter to only include those saved for a specific query id.
        with_ref: sUnion[str, None] = None
            Filter to only include those saved for a specific reference id.

        Returns
        -------
        List[str]:
            List of query -> reference pairs matching search criteria.
        """
        pairs = [
            [q_id, ref_id]
            for q_id in self._db.keys()
            for ref_id in self._db[q_id].keys()
        ]
        pairs = list(filter(lambda p: p[0] == with_q or with_q is None, pairs))
        pairs = list(filter(lambda p: p[1] == with_ref or with_ref is None, pairs))
        pairs = [" -> ".join(p) for p in pairs]
        return pairs
