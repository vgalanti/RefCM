import os
import io
import logging
import warnings
import contextlib
import numpy as np
import scanpy as sc
import rpy2.robjects as ro
import rpy2.rinterface_lib.callbacks as cb

from anndata import AnnData
from scipy.sparse import issparse, csc_matrix
from IPython.utils.io import capture_output

""" --------------- stderr redirect ---------------"""


@contextlib.contextmanager
def suppress_all_console():

    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    old_filters = warnings.filters[:]
    warnings.simplefilter("ignore")

    with contextlib.ExitStack() as stack:
        stack.enter_context(capture_output())
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

        try:
            yield

        finally:
            logging.disable(old_disable)
            warnings.filters[:] = old_filters


class suppress_r_console:
    def __enter__(self):
        self._old_write = cb.consolewrite_print
        self._old_warn = cb.consolewrite_warnerror
        cb.consolewrite_print = lambda _: None
        cb.consolewrite_warnerror = lambda _: None

    def __exit__(self, *_):
        cb.consolewrite_print = self._old_write
        cb.consolewrite_warnerror = self._old_warn


""" --------------- time formatting ---------------"""


def fmt_time(seconds: float) -> str:
    s = float(seconds)
    if s < 60:
        return f"{s:0.3f}s"

    m, rem = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{rem:05.2f}s"

    h, rem = divmod(m, 60)
    return f"{int(h)}h{int(rem):02d}m{(s % 60):05.2f}s"


""" --------------- preprocessing ---------------"""


def prep_adata(
    adata: AnnData, n_hvg: int = 2000, target_sum: float | None = None
) -> AnnData:

    adata.obs_names_make_unique()

    if "counts" not in adata.layers:
        X = adata.X
        adata.layers["counts"] = X.tocsc() if issparse(X) else csc_matrix(X)

    adata.X = adata.layers["counts"].copy()
    if target_sum is None or target_sum > 0:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)

    return adata


def load_adata(fpath: str) -> AnnData:
    adata = sc.read_h5ad(fpath)
    prep_adata(adata)
    return adata


def prep_umap(adata: AnnData) -> AnnData:
    # coarse check that prep_adata had already been run
    assert "highly_variable" in adata.var

    if "umap" in adata.uns:
        return adata

    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    return adata


""" --------------- R utils ---------------"""


with suppress_r_console():
    ro.r(
        """
        library(Matrix)
        library(dplyr)
        library(Seurat)
        library(SingleCellExperiment)
        library(CIPR)
        library(SingleR)
        library(scmap)
        library(clustifyr)
        """
    )


def adata_to_r_dgCMatrix(adata: AnnData, counts: bool = False, hvg: bool = True):

    mask = adata.var["highly_variable"].to_numpy() if hvg else slice(None)

    genes = adata.var_names[mask].astype(str).tolist()
    cells = adata.obs_names.astype(str).tolist()

    Xsrc = adata.layers["counts"] if counts else adata.X
    X = Xsrc[:, mask].T.tocsc()
    X.sort_indices()

    return ro.r["new"](
        "dgCMatrix",
        Dim=ro.IntVector([X.shape[0], X.shape[1]]),
        x=ro.FloatVector(np.asarray(X.data, dtype=np.float64)),
        i=ro.IntVector(np.asarray(X.indices, dtype=np.int32)),
        p=ro.IntVector(np.asarray(X.indptr, dtype=np.int32)),
        Dimnames=ro.r["list"](ro.StrVector(genes), ro.StrVector(cells)),
    )


def obs_to_r_strvec(adata, obs_col: str):
    return ro.StrVector(adata.obs[obs_col].astype(str).tolist())


def obs_to_r_factor(adata: AnnData, obs_col: str):
    return ro.r["factor"](obs_to_r_strvec(adata, obs_col))


def obs_to_r_namedstrvec(adata: AnnData, obs_col: str):
    v = ro.StrVector(adata.obs[obs_col].astype(str).tolist())
    v.names = ro.StrVector(adata.obs_names.astype(str).tolist())
    return v
