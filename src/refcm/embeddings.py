import logging
import numpy as np
import scanpy as sc
import scipy.sparse

from abc import ABC, abstractmethod
from typing import Literal
from anndata import AnnData
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

MAX_CLUSTER_SIZE = 15_000


class Embedder(ABC):
    """
    Base class for RefCM-compatible embeddings.

    Expects log-normalized expression in .X.
    """

    @abstractmethod
    def fit(self, q: AnnData, ref: AnnData) -> None:
        """Fit to a query and reference pair."""
        pass

    @abstractmethod
    def embed(self, cluster: AnnData) -> np.ndarray:
        """Return embedding for a cluster."""
        pass

    @staticmethod
    def _check_log_normalized(adata: AnnData, name: str) -> None:
        """Warn if .X appears to contain counts rather than log-normalized data."""
        sample = adata.X[:100, :100]
        if scipy.sparse.issparse(sample):
            sample = sample.toarray()

        if np.allclose(sample, sample.astype(int)):
            log.warning(
                f"{name}.X appears to contain counts. Expected log-normalized expression."
            )

    @staticmethod
    def _subsample(x: np.ndarray, max_size: int | None) -> np.ndarray:
        """Subsample array if it exceeds max_size."""
        if max_size is not None and x.shape[0] > max_size:
            idx = np.random.choice(x.shape[0], max_size, replace=False)
            return x[idx]
        return x

    @staticmethod
    def _to_dense(x) -> np.ndarray:
        """Convert sparse matrix to dense array if needed."""
        if scipy.sparse.issparse(x):
            return x.toarray()
        return x


class HVGEmbedder(Embedder):
    """
    Joint Highly Variable Gene embedding.

    Selects HVGs from both query and reference, takes their union
    intersected with common genes, and uses raw expression values.
    """

    def __init__(
        self,
        n_hvg: int = 1200,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
    ) -> None:
        """
        Initialize HVG embedder.

        Parameters
        ----------
        n_hvg
            Number of highly variable genes to select per dataset.
        max_cluster_size
            Maximum cluster size (larger clusters are subsampled).
        """
        self._n_hvg = n_hvg
        self._max_cluster_size = max_cluster_size
        self._genes: list[str] = []

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """Fit to query and reference, computing joint HVG set."""
        self._check_log_normalized(q, "query")
        self._check_log_normalized(ref, "reference")

        hvg_q = self._compute_hvg(q)
        hvg_ref = self._compute_hvg(ref)
        common = np.intersect1d(q.var_names, ref.var_names)

        self._genes = list((set(hvg_q) | set(hvg_ref)) & set(common))
        log.debug(f"Using {len(self._genes)} genes.")

    def embed(self, cluster: AnnData) -> np.ndarray:
        """Return gene-subset expression matrix."""
        x = cluster[:, self._genes].X
        x = self._to_dense(x)
        return self._subsample(x, self._max_cluster_size)

    def _compute_hvg(self, adata: AnnData) -> list[str]:
        """Compute highly variable genes."""
        sc.pp.highly_variable_genes(adata, n_top_genes=self._n_hvg)
        return adata.var_names[adata.var["highly_variable"]].tolist()


class ScikitEmbedder(Embedder):
    """
    Dimensionality reduction embedding using scikit-learn.

    Supports PCA, ICA, and NMF. Fits on concatenated query and reference,
    then transforms individual clusters.
    """

    def __init__(
        self,
        method: Literal["PCA", "ICA", "NMF"] = "PCA",
        n_components: int = 100,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
        scale: bool = False,
    ) -> None:
        """
        Initialize scikit-learn based embedder.

        Parameters
        ----------
        method
            Dimensionality reduction method.
        n_components
            Number of components to retain.
        max_cluster_size
            Maximum cluster size (larger clusters are subsampled).
        scale
            Whether to standardize features before transformation (disabled for NMF).
        """
        self._method_name = method
        self._n_components = n_components
        self._max_cluster_size = max_cluster_size
        self._scale = scale and method != "NMF"
        self._scaler: StandardScaler | None = None
        self._genes: np.ndarray = np.array([])

        match method:
            case "PCA":
                self._model = PCA(n_components, random_state=42)
            case "ICA":
                self._model = FastICA(n_components, random_state=42, max_iter=500)
            case "NMF" | "NNMF":
                self._model = NMF(n_components, random_state=42)
            case _:
                raise ValueError(f"Unknown embedding method: {method}")

    def fit(self, q: AnnData, ref: AnnData) -> None:
        """Fit to query and reference datasets."""
        self._check_log_normalized(q, "query")
        self._check_log_normalized(ref, "reference")

        self._genes = np.intersect1d(q.var_names, ref.var_names)

        q_x = self._to_dense(q[:, self._genes].X)
        ref_x = self._to_dense(ref[:, self._genes].X)
        x = np.concatenate((q_x, ref_x))

        if self._scale:
            self._scaler = StandardScaler()
            x = self._scaler.fit_transform(x)

        self._model.fit(x)
        log.debug(f"{self._method_name} (n={self._n_components}) fitting complete.")

    def embed(self, cluster: AnnData) -> np.ndarray:
        """Return embedded expression matrix."""
        x = self._to_dense(cluster[:, self._genes].X)

        if self._scale and self._scaler is not None:
            x = self._scaler.transform(x)

        x = self._model.transform(x)
        return self._subsample(x, self._max_cluster_size)


class PCAEmbedder(ScikitEmbedder):
    """PCA-based embedding."""

    def __init__(
        self,
        n_components: int = 200,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
        scale: bool = False,
    ) -> None:
        super().__init__(
            method="PCA",
            n_components=n_components,
            max_cluster_size=max_cluster_size,
            scale=scale,
        )


class ICAEmbedder(ScikitEmbedder):
    """ICA-based embedding."""

    def __init__(
        self,
        n_components: int = 100,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
        scale: bool = False,
    ) -> None:
        super().__init__(
            method="ICA",
            n_components=n_components,
            max_cluster_size=max_cluster_size,
            scale=scale,
        )


class NMFEmbedder(ScikitEmbedder):
    """Non-negative Matrix Factorization embedding."""

    def __init__(
        self,
        n_components: int = 100,
        max_cluster_size: int | None = MAX_CLUSTER_SIZE,
    ) -> None:
        super().__init__(
            method="NMF",
            n_components=n_components,
            max_cluster_size=max_cluster_size,
            scale=False,
        )
