import json
import scipy
import logging
import numpy as np
import scanpy as sc

from typing import List, Literal
from hashlib import sha256
from anndata import AnnData
from sklearn.decomposition import PCA, FastICA, NMF

log = logging.getLogger(__name__)

MAX_CLUSTER_SIZE = 6000
MAX_HVG_SIZE = 30000


class Embedder:
    """
    refcm-compatible embedding.
    """

    uid: str

    def __init__(self) -> None:
        pass

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """Fits to a given query and reference pair."""
        pass

    def embed(self, cluster: AnnData) -> np.ndarray:
        """Returns the embedding for a given cluster."""
        pass


class HVGEmbedder(Embedder):
    def __init__(
        self,
        n_top_genes: int = 1200,
        target_sum: int | None = None,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
        max_hvg_size: int | None = MAX_HVG_SIZE,
    ) -> None:
        """
        Creates a Joint HVG Embedding object as described in the paper.

        n_top_genes: int = 1200
            Number of highly variable genes to select per dataset.
        target_sum: int | None = None
            Target row sum after log-normalization.
        max_cluster_size: int | None = MAX_CLUSTER_SIZE
            The maximum cluster size we subsample larger clusters to.
        max_hvg_size: int | None = MAX_HVG_SIZE
            The maximum number of rows we pass to scanpy's hvg computation.

        """
        super().__init__()

        self._n_top_genes: int = n_top_genes
        self._target_sum: int | None = target_sum
        self._max_cluster_size: int | None = max_cluster_size
        self._max_hvg_size: int | None = max_hvg_size

        uid = {
            "n_top_genes": self._n_top_genes,
            "target_sum": self._target_sum,
            "max_cluster_size": self._max_cluster_size,
            "max_hvg_size": self._max_hvg_size,
        }
        uid = json.dumps(uid).encode("utf-8")
        self.uid = sha256(uid).hexdigest()

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """
        Fits to the given query and reference datasets.

        Parameters
        ----------
        q: AnnData
            Query dataset.
        ref: AnnData
            Reference dataset.
        """
        self.q = q
        self.ref = ref

        # apply basic pre-processing
        self._preprocess()

        # determine geneset gs to utilize
        hvg = set(self._hvg(q)).union(self._hvg(ref))
        common = np.intersect1d(q.var_names, ref.var_names)
        self.gs = list(set(hvg) & set(common))

        log.debug(f"Using {len(self.gs)} genes.")

        # clean up changes done to q and ref
        self._cleanup()

    def _preprocess(self) -> None:
        """
        Applies basic pre-processing (log/count-normalization).
        Operations done in-place for memory constraints, undone in ._cleanup().
        """
        # log1p normalize query and reference
        self.q_sums = self.q.X.sum(axis=1).reshape((-1, 1))
        self.ref_sums = self.ref.X.sum(axis=1).reshape((-1, 1))

        if self._target_sum is not None:
            self.q.X = self._target_sum * self.q.X / self.q_sums
            self.ref.X = self._target_sum * self.ref.X / self.ref_sums

        self.q.X = np.log1p(self.q.X)
        self.ref.X = np.log1p(self.ref.X)

    def _cleanup(self) -> None:
        """Cleans up changes done to q and ref during init."""
        self.q.X = np.expm1(self.q.X)
        self.ref.X = np.expm1(self.ref.X)

        if self._target_sum is not None:
            self.q.X *= self.q_sums / self._target_sum
            self.ref.X *= self.ref_sums / self._target_sum

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
        if self._max_hvg_size is None:
            tmp = ds.copy()
        else:
            n = ds.X.shape[0]
            sbset = np.random.choice(n, min(n, self._max_hvg_size), replace=False)
            tmp = ds[sbset].copy()

        sc.pp.highly_variable_genes(tmp, n_top_genes=self._n_top_genes)

        return tmp.var["highly_variable"][tmp.var["highly_variable"]].index.to_list()

    def embed(self, cluster: AnnData) -> np.ndarray:
        """
        Returns the embedding for a given cluster.

        Parameters
        ----------
        cluster: AnnData
            The AnnData for the cluster to embed.

        Returns
        -------
        np.ndarray
            The embedding for said cluster.
        """
        # gene subset
        x = cluster[:, self.gs].X.copy()

        # log/count-normalization
        if self._target_sum is not None:
            x *= self._target_sum / x.sum(axis=1).reshape((-1, 1))
        x = np.log1p(x)

        # memory regulation
        if self._max_cluster_size is not None and x.shape[0] > self._max_cluster_size:
            sbs = np.random.choice(x.shape[0], self._max_cluster_size, replace=False)
            x = x[sbs]

        # conversion to full matrix, if sparse
        if scipy.sparse.issparse(x):
            x = x.toarray()

        return x


class ScikitEmbedder(Embedder):
    def __init__(
        self,
        method: Literal["PCA", "ICA", "NMF"] = "PCA",
        n_components: int = 100,
        log_norm: bool = True,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
    ) -> None:
        """
        Creates a PCA, ICA, or NMF embedder using Scikit-Learn's implementations.

        n_components: int = 100
            Number of components to choose.
        log_norm: bool = True
            Whether to log-normalize the raw counts.
        max_cluster_size: int | None = MAX_CLUSTER_SIZE
            The maximum cluster size we subsample larger clusters to.
        """
        super().__init__()
        self.mt: str = method
        match method:
            case "PCA":
                self.method = PCA(n_components)
            case "ICA":
                self.method = FastICA(n_components)
            case "NMF" | "NNMF":
                self.method = NMF(n_components)
            case _:
                log.error(f"Embedding method {method} invalid.")
                return

        self._n_components: int = n_components
        self._log_norm: bool = log_norm
        self._max_cluster_size: int | None = max_cluster_size

        uid = {
            "method": self.mt,
            "n_components": self._n_components,
            "log_norm": self._log_norm,
            "max_cluster_size": self._max_cluster_size,
        }
        uid = json.dumps(uid).encode("utf-8")
        self.uid = sha256(uid).hexdigest()

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """
        Fits to the given query and reference datasets.

        Parameters
        ----------
        q: AnnData
            Query dataset.
        ref: AnnData
            Reference dataset.
        """
        self.q = q
        self.ref = ref

        # apply basic pre-processing
        if self._log_norm:
            self.q.X = np.log1p(self.q.X)
            self.ref.X = np.log1p(self.ref.X)

        # method fitting
        self.gs = np.intersect1d(q.var_names, ref.var_names)
        self.method.fit(np.concatenate((q[:, self.gs].X, ref[:, self.gs].X)))
        log.debug(f"{self.mt} (n={self._n_components}) fitting complete.")

        # clean up changes done to q and ref
        if self._log_norm:
            self.q.X = np.expm1(self.q.X)
            self.ref.X = np.expm1(self.ref.X)

    def embed(self, cluster: AnnData) -> np.ndarray:
        """
        Returns the embedding for a given cluster.

        Parameters
        ----------
        cluster: AnnData
            The AnnData for the cluster to embed.

        Returns
        -------
        np.ndarray
            The embedding for said cluster.

        """
        # gene subset
        x = cluster[:, self.gs].X.copy()

        # log/count-normalization
        if self._log_norm:
            x = np.log1p(x)

        # apply PCA reduction
        x = self.method.transform(x)

        # memory regulation
        if self._max_cluster_size is not None and x.shape[0] > self._max_cluster_size:
            sbs = np.random.choice(x.shape[0], self._max_cluster_size, replace=False)
            x = x[sbs]

        # conversion to full matrix, if sparse
        if scipy.sparse.issparse(x):
            x = x.toarray()

        return x


class PCAEmbedder(ScikitEmbedder):
    def __init__(
        self,
        n_components: int = 200,
        log_norm: bool = True,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
    ) -> None:
        super().__init__(
            method="PCA",
            n_components=n_components,
            log_norm=log_norm,
            max_cluster_size=max_cluster_size,
        )


class ICAEmbedder(ScikitEmbedder):
    def __init__(
        self,
        n_components: int = 100,
        log_norm: bool = True,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
    ) -> None:
        super().__init__(
            method="ICA",
            n_components=n_components,
            log_norm=log_norm,
            max_cluster_size=max_cluster_size,
        )


class NMFEmbedder(ScikitEmbedder):
    def __init__(
        self,
        n_components: int = 200,
        log_norm: bool = True,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
    ) -> None:
        super().__init__(
            method="NMF",
            n_components=n_components,
            log_norm=log_norm,
            max_cluster_size=max_cluster_size,
        )
