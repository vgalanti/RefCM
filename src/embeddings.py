import abc
import logging
import scipy
import numpy as np
import pandas as pd
import scanpy as sc

from typing import List
from anndata import AnnData
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

MAX_CLUSTER_SIZE = 6000
MAX_HVG_SIZE = 30000
MAX_PCA_SIZE = 50000

# TODO add an Embedding field into the cache to account for these new embedding methods
# TODO add the marker gene options
# TODO PCA
# TODO ICA
# TODO NNMF
# TODO VAE
# TODO scGCN


class Embedder:
    """
    refcm-compatible embedding.
    """

    id_: str

    def __init__(self) -> None:
        pass

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """Fits to a given query and reference pair."""

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
            self.ef.X = self._target_sum * self.ref.X / self.ref_sums

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
            sbset = np.random.choice(n, min(n, MAX_HVG_SIZE), replace=False)
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

        TODO
        ----
        * not sure we need the target sum rescaling,
          in theory this shouldn't affect relative WS costs downstream.
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


class PCAEmbedder(Embedder):

    def __init__(
        self,
        n_components: int = 100,
        log_norm: bool = True,
        target_sum: int | None = None,
        max_cluster_size: int | None = None,
        # max_pca_size: int | None = MAX_HVG_SIZE,
    ) -> None:
        """
        Creates a Joint HVG Embedding object as described in the paper.

        n_top_genes: int = 1200
            Number of highly variable genes to select per dataset.
        target_sum: int | None = None
            Target row sum after log-normalization.
        max_cluster_size: int | None = MAX_CLUSTER_SIZE
            The maximum cluster size we subsample larger clusters to.
        max_pca_size: int | None = MAX_HVG_SIZE
            The maximum number of rows we pass to scanpy's hvg computation.

        """
        super().__init__()

        self.pca = PCA(n_components)
        self._n_components: int = n_components
        self._log_norm: bool = log_norm
        self._target_sum: int | None = target_sum
        self._max_cluster_size: int | None = max_cluster_size
        # self._max_pca_size: int | None = max_pca_size

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """
        Fits to the given query and reference datasets.

        Parameters
        ----------
        q: AnnData
            Query dataset.
        ref: AnnData
            Reference dataset.

        TODO
        ----
        * determine if we want to fit PCA on both query and reference,
        as in concatenate them together.
        """
        self.q = q
        self.ref = ref

        # apply basic pre-processing
        self._preprocess()

        # PCA fitting
        self.gs = np.intersect1d(q.var_names, ref.var_names)
        self.pca.fit(np.concatenate((q[:, self.gs].X, ref[:, self.gs].X)))
        log.debug(f"PCA fitting with {self._n_components} complete.")

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
            self.ef.X = self._target_sum * self.ref.X / self.ref_sums

        if self._log_norm:
            self.q.X = np.log1p(self.q.X)
            self.ref.X = np.log1p(self.ref.X)

    def _cleanup(self) -> None:
        """Cleans up changes done to q and ref during init."""
        if self._log_norm:
            self.q.X = np.expm1(self.q.X)
            self.ref.X = np.expm1(self.ref.X)

        if self._target_sum is not None:
            self.q.X *= self.q_sums / self._target_sum
            self.ref.X *= self.ref_sums / self._target_sum

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

        TODO
        ----
        * not sure we need the target sum rescaling,
          in theory this shouldn't affect relative WS costs downstream.
        """
        # gene subset
        x = cluster[:, self.gs].X.copy()

        # log/count-normalization
        if self._target_sum is not None:
            x *= self._target_sum / x.sum(axis=1).reshape((-1, 1))

        if self._log_norm:
            x = np.log1p(x)

        # apply PCA reduction
        x = self.pca.transform(x)

        # memory regulation
        if self._max_cluster_size is not None and x.shape[0] > self._max_cluster_size:
            sbs = np.random.choice(x.shape[0], self._max_cluster_size, replace=False)
            x = x[sbs]

        # conversion to full matrix, if sparse
        if scipy.sparse.issparse(x):
            x = x.toarray()

        return x