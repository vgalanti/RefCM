import os
import ot
import logging
import warnings
import pulp as pl
import numpy as np

from tqdm import tqdm
from numpy import ndarray
from typing import Callable, Literal
from anndata import AnnData
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import pairwise_distances

from .embeddings import Embedder, HVGEmbedder

log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)


class RefCM:
    """
    Reference-based Cell type Matching using Optimal Transport.

    Maps query cell clusters to reference cell types by computing
    Wasserstein distances and solving an integer programming matching problem.
    """

    def __init__(
        self,
        pdist: str = "euclidean",
        embedder: Embedder = HVGEmbedder(),
        max_merges: int = 1,
        max_splits: int = 1,
        n_target_clusters: int | None = None,
        discovery_threshold: float | None = 0.1,
        ot_solver: Literal["sink", "emd", "gw", "uot"] = "emd",
        reg: float = 0.05,
        num_iter_max: int = int(1e6),
        uot_weight_fn: (
            Callable[[ndarray, ndarray], tuple[ndarray, ndarray]] | None
        ) = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize RefCM instance.

        Parameters
        ----------
        pdist
            Pairwise distance metric (compatible with sklearn's pairwise_distances).
        embedder
            Embedding strategy for cells.
        max_merges
            Maximum query clusters that can merge to a single reference cluster.
            Use -1 for unlimited.
        max_splits
            Maximum reference clusters a single query cluster can map to.
            Use -1 for unlimited.
        n_target_clusters
            Maximum number of reference clusters to map to.
        discovery_threshold
            Fraction (0-1) of worst-matching costs to reject as potential novel types.
            E.g., 0.1 rejects the worst 10% of matches.
        ot_solver
            Optimal transport solver: "sink" (Sinkhorn), "emd" (Earth Mover's Distance),
            "gw" (Gromov-Wasserstein), or "uot" (Unbalanced OT).
        reg
            Regularization parameter for Sinkhorn (applied as reg * median(distances)).
        num_iter_max
            Maximum iterations for OT optimization.
        uot_weight_fn
            For unbalanced OT: function(query_expr, ref_expr) -> (query_weights, ref_weights).
        verbose
            Show progress bar during computation.
        """
        # embed
        self._pdist = pdist
        self._embedder = embedder

        # ip
        self._max_merges = max_merges
        self._max_splits = max_splits
        self._n_target_clusters = n_target_clusters
        self._discovery_threshold = discovery_threshold

        # ot
        self._ot_solver = ot_solver
        self._reg = reg
        self._num_iter_max = num_iter_max
        self._uot_weight_fn = uot_weight_fn

        # ref
        self._ref: AnnData | None = None
        self._ref_key: str | None = None
        self._ref_ltk: dict = {}
        self._ref_ktl: dict = {}
        self._ref_nc: int = 0

        self._verbose = verbose

    def setref(self, ref: AnnData, ref_key: str) -> None:
        """
        Set the reference dataset for annotation.

        Parameters
        ----------
        ref
            Reference dataset with ground truth labels.
        ref_key
            Column in ref.obs containing cell type labels.
        """
        if ref_key not in ref.obs.columns:
            raise KeyError(f"Reference .obs does not contain '{ref_key}'")

        labels = np.sort(np.unique(ref.obs[ref_key]))

        self._ref = ref
        self._ref_key = ref_key
        self._ref_nc = len(labels)

        # label-to-key and and key-to-label maps
        self._ref_ltk = {label: i for i, label in enumerate(labels)}
        self._ref_ktl = {i: label for i, label in enumerate(labels)}

    def annotate(
        self,
        q: AnnData,
        q_key: str,
        use_cached: bool = False,
    ) -> AnnData:
        """
        Annotate a query dataset.

        Parameters
        ----------
        q
            Query dataset to annotate.
        q_key
            Column in q.obs containing cluster assignments.
        use_cached
            If True, reuse cached costs from q.uns["refcm"] if dimensions match.

        Returns
        -------
        AnnData
            The query dataset with annotations in q.obs["refcm"] and
            metadata in q.uns["refcm"].
        """
        if self._ref is None:
            raise RuntimeError("Reference not set. Call .setref() before .annotate()")

        if q_key not in q.obs.columns:
            raise KeyError(f"Query key '{q_key!r}' not found in q.obs")

        clusters = np.sort(np.unique(q.obs[q_key]))
        q_nc = len(clusters)
        c2i = {label: i for i, label in enumerate(clusters)}

        # attempt cache
        if use_cached:
            cached = self._load(q, q_nc)
            if cached is not None:
                cost, m = cached
                self._label(q, q_key, c2i, m)
                return q

        # compute fresh
        cost = self._ws(q, q_key, c2i, q_nc)
        cost = self._norm(cost)
        m = self._ip(self._thresh(cost))

        self._label(q, q_key, c2i, m)
        self._save(q, cost, m)
        return q

    def _load(self, q: AnnData, q_nc: int) -> tuple[ndarray, ndarray] | None:
        """Load cached costs and mapping from q.uns."""
        if "refcm" not in q.uns:
            log.error(
                "use_cached=True but q.uns['refcm'] not found. Rerun with use_cached=False."
            )
            return None

        cache = q.uns["refcm"]
        if "costs" not in cache or "mapping" not in cache:
            log.error(
                "use_cached=True but cached data incomplete. Rerun with use_cached=False."
            )
            return None

        cost, m = cache["costs"], cache["mapping"]
        if cost.shape != (q_nc, self._ref_nc):
            log.error(
                f"use_cached=True but dimension mismatch: cached {cost.shape}, "
                f"expected ({q_nc}, {self._ref_nc}). Rerun with use_cached=False."
            )
            return None

        log.info("Using cached costs and mapping.")
        return cost, m

    def _norm(self, cost: ndarray) -> ndarray:
        """Normalize cost matrix to [-1, 0]."""
        mx, mn = cost.max(), cost.min()
        if mx == mn:
            return -np.ones_like(cost)
        return (cost - mx) / (mx - mn)

    def _thresh(self, cost: ndarray) -> ndarray:
        """Apply discovery threshold."""
        if self._discovery_threshold is None:
            return cost.copy()
        c = cost.copy()
        t = np.percentile(c, (1 - self._discovery_threshold) * 100)
        c[c >= t] = 1e2
        return c

    def _label(self, q: AnnData, q_key: str, c2i: dict, m: ndarray) -> None:
        """Apply annotations to query."""
        q.obs["refcm"] = ""
        for label, idx in c2i.items():
            mapped = np.argmax(m[idx] == 1)
            annot = "novel" if m[idx][mapped] == 0 else self._ref_ktl[mapped]
            q.obs.loc[q.obs[q_key] == label, "refcm"] = annot

    def _save(self, q: AnnData, cost: ndarray, m: ndarray) -> None:
        """Save results to q.uns['refcm']."""
        q.uns["refcm"] = {"costs": cost, "mapping": m, "ref_ktl": self._ref_ktl.copy()}

    def _ws(self, q: AnnData, q_key: str, c2i: dict, q_nc: int) -> ndarray:
        """Compute Wasserstein distances between query and reference clusters."""
        self._embedder.fit(q, self._ref)

        if self._ot_solver == "uot" and self._uot_weight_fn is None:
            log.warning(
                "Unbalanced OT selected but no weight function provided. Using uniform."
            )

        # precompute integer indices (faster than boolean masking)
        q_idx = {idx: np.where(q.obs[q_key] == lab)[0] for lab, idx in c2i.items()}
        r_idx = {
            rc: np.where(self._ref.obs[self._ref_key] == self._ref_ktl[rc])[0]
            for rc in range(self._ref_nc)
        }

        q_emb = {idx: self._embedder.embed(q[q_idx[idx]]) for idx in range(q_nc)}
        r_emb = {
            rc: self._embedder.embed(self._ref[r_idx[rc]]) for rc in range(self._ref_nc)
        }

        # parallel OT
        ws = np.zeros((q_nc, self._ref_nc))
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            futures = [
                pool.submit(self._ot, q_emb[qc], r_emb[rc], qc, rc)
                for qc in range(q_nc)
                for rc in range(self._ref_nc)
            ]
            fmt = "|{bar:16}| [{percentage:>6.2f}%] : {elapsed}"
            with tqdm(
                total=len(futures), bar_format=fmt, disable=not self._verbose
            ) as pbar:
                for future in as_completed(futures):
                    qc, rc, cost = future.result()
                    ws[qc, rc] = cost
                    pbar.update(1)
        return ws

    def _ot(
        self, x_q: ndarray, x_r: ndarray, qc: int, rc: int
    ) -> tuple[int, int, float]:
        """Compute OT cost for a cluster pair."""
        a, b = np.ones(len(x_q)) / len(x_q), np.ones(len(x_r)) / len(x_r)

        match self._ot_solver:
            case "sink":
                M = self._dist(x_q, x_r)
                cost = ot.sinkhorn2(
                    a, b, M, reg=self._reg * np.median(M), numItermax=self._num_iter_max
                )
            case "emd":
                M = self._dist(x_q, x_r)
                cost = ot.emd2(a, b, M, numItermax=self._num_iter_max)
            case "gw":
                C1, C2 = self._dist(x_q, x_q), self._dist(x_r, x_r)
                cost = ot.gromov_wasserstein2(C1, C2, a, b)
            case "uot":
                M = self._dist(x_q, x_r)
                reg = self._reg * np.median(M)
                if self._uot_weight_fn is not None:
                    a, b = self._uot_weight_fn(x_q, x_r)
                cost = ot.sinkhorn_unbalanced2(
                    a, b, M, reg, [reg], numItermax=self._num_iter_max
                )
            case _:
                raise ValueError(f"Unknown OT solver: {self._ot_solver}")
        return qc, rc, cost

    def _dist(self, a: ndarray, b: ndarray) -> ndarray:
        """Pairwise distance matrix."""
        if self._pdist in ("inner", "dot"):
            return -a @ b.T
        return pairwise_distances(a, b, self._pdist)

    def _ip(self, cost: ndarray) -> ndarray:
        """Solve integer programming matching problem."""
        prob = pl.LpProblem()
        q_nc, ref_nc = cost.shape
        qi, ri = np.arange(q_nc), np.arange(ref_nc)

        e = pl.LpVariable.dicts("e", (qi, ri), 0, 1, pl.LpInteger)
        prob += pl.lpSum(cost[u][v] * e[u][v] for u in qi for v in ri)

        if self._max_splits >= 0:
            for u in qi:
                prob += pl.lpSum(e[u][v] for v in ri) <= self._max_splits

        if self._max_merges >= 0:
            for v in ri:
                prob += pl.lpSum(e[u][v] for u in qi) <= self._max_merges

        if self._n_target_clusters is not None:
            w = pl.LpVariable.dicts("w", ri, 0, 1, pl.LpInteger)
            for i in ri:
                prob += w[i] <= pl.lpSum(e[v][i] for v in qi)
                prob += pl.lpSum(e[v][i] for v in qi) <= w[i] * len(qi)
            prob += pl.lpSum(w[i] for i in ri) <= self._n_target_clusters

        prob.solve(pl.GLPK_CMD(msg=False))
        log.debug(f"IP status: {pl.LpStatus[prob.status]}")

        m = np.zeros_like(cost)
        for v in prob.variables():
            if v.name.startswith("e_"):
                i, j = (int(x) for x in v.name.split("_")[1:])
                m[i, j] = v.varValue
        return m
