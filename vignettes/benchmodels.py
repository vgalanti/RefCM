import sys

sys.path.append("../src/")

import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd

from time import perf_counter
from anndata import AnnData

# benchdb
from benchdb import Annot
from benchutils import suppress_all_console, suppress_r_console, fmt_time
from benchutils import obs_to_r_factor, obs_to_r_namedstrvec, adata_to_r_dgCMatrix

# benchmark model imports
import scvi
import celltypist
import rpy2.robjects as ro

from refcm import RefCM as _RCM
from sklearn.svm import LinearSVC
from scalex import SCALEX as _SCALEX, label_transfer

# config
sc.settings.verbosity = 0
scvi.settings.seed = 0
torch.set_float32_matmul_precision("high")

""" --------------- constants ---------------"""

RCM_METHODS = ["RefCM", "CIPR", "SingleRcluster", "clustifyr"]
RM_METHODS = [
    "SVM",
    "CellTypist",
    "Seurat",
    "SCALEX",
    "scANVI",
    "SingleR",
    "scmapcell",
    "scmapcluster",
]
MV_METHODS = [
    "MV-SVM",
    "MV-CellTypist",
    "MV-Seurat",
    "MV-SCALEX",
    "MV-scANVI",
    "MV-SingleR",
    "MV-scmapcell",
    "MV-scmapcluster",
]


""" --------------- BenchModel ---------------"""


class BenchModel:
    """
    A common interface for all benchmarked methods, as
    reference mapping and reference cluster mapping algorithms.
    """

    id_: str

    def __init__(self) -> None:
        # perf timing variables
        self._t_ref_s = None
        self._t_ref_e = None
        self._t_ref = None

        self._t_annot_s = None
        self._t_annot_e = None
        self._t_annot = None
        self._t_total = None

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:
        """
        Set the reference dataset reference dataset.

        Parameters
        ----------
        ref: AnnData
            The reference dataset with raw counts in .X
        rkey: str
            .obs key for reference labels.
        """
        self.ref = ref
        self.rkey = rkey

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> Annot:
        """
        Annotates the query dataset using the reference data, and adds
        its predictions as a new column (self.id) under the query's .obs.

        Parameters
        ----------
        q: AnnData
            The query dataset with raw counts in .X
        qkey: str
            The query .obs key with the clustering information

        Returns
        -------
        list: list of predictions
        """
        self.q = q
        self.qkey = qkey

        return Annot()

    def _t_ref_start(self) -> None:
        self._t_ref_s = perf_counter()

    def _t_ref_end(self) -> None:
        self._t_ref_e = perf_counter()
        self._t_ref = self._t_ref_e - self._t_ref_s

    def _t_annot_start(self) -> None:
        self._t_annot_s = perf_counter()

    def _t_annot_end(self) -> None:
        self._t_annot_e = perf_counter()
        self._t_annot = self._t_annot_e - self._t_annot_s
        self._t_total = self._t_annot + (self._t_ref if self._t_ref is not None else 0)

    @property
    def time(self) -> str:
        rtime = fmt_time(self._t_ref) if self._t_ref is not None else "N/A"
        atime = fmt_time(self._t_annot) if self._t_annot is not None else "N/A"
        ttime = fmt_time(self._t_total) if self._t_total is not None else "N/A"
        return f"time: {ttime:>12} | setref: {rtime:>12} | annot: {atime:>12}"


""" --------------- Seurat ---------------"""


class Seurat(BenchModel):
    def __init__(self) -> None:
        self.id_ = "Seurat"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # pass ref to R
        self._to_r(ref, "ref")
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey

        # pass q to R
        self._to_r(q, "q")

        # run seurat
        self._t_annot_start()
        with suppress_r_console():
            ro.r(
                """
                anchors <- FindTransferAnchors(reference = ref, query = q, dims = 1:30)
                
                # catch exception where < default 50 anchors are available
                n_anchor_cells <- length(unique(anchors@anchors[, 2])) 
                kW <- min(50, max(5, n_anchor_cells - 1))

                # ref_labels is a named vector over ref cells
                preds <- TransferData(
                    anchorset = anchors,
                    refdata = ref_labels,
                    dims = 1:30,
                    k.weight = kW
                )

                q <- AddMetaData(object = q, metadata = preds)
                preds <- q$predicted.id
                """
            )
        self._t_annot_end()

        preds = list(ro.globalenv["preds"])

        # return annotations
        return Annot(preds, self.ref, self.rkey)

    def _to_r(self, adata: AnnData, name: str):

        ro.globalenv["counts"] = adata_to_r_dgCMatrix(adata, counts=True, hvg=False)

        with suppress_r_console():
            ro.r(
                f"""
                {name} <- CreateSeuratObject(counts = counts)
                {name} <- NormalizeData({name}, verbose = FALSE)
                {name} <- FindVariableFeatures({name}, selection.method = "vst", nfeatures = 2000)
                """
            )


""" --------------- CIPR ---------------"""


class CIPR(BenchModel):
    def __init__(self) -> None:
        self.id_ = "CIPR"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # pass ref to R
        self._to_r(ref, "ref")
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)

        # do additional CIPR-required ref processing
        self._t_ref_start()
        ro.r(
            """
            ref$ref_label <- ref_labels
            Idents(ref) <- "ref_label"

            # CIPR custom reference: data.frame with column 'gene' + one column per reference label
            ref_avg <- AverageExpression(ref, assays = "RNA")$RNA
            ref_avg <- as.data.frame(ref_avg)
            ref_avg$gene <- rownames(ref_avg)

            # Minimal custom_ref_annot (optional but keeps CIPR happy / informative)
            ref_annot <- data.frame(
                short_name = setdiff(colnames(ref_avg), "gene"),
                long_name = setdiff(colnames(ref_avg), "gene"),
                description = "",
                reference_cell_type = setdiff(colnames(ref_avg), "gene"),
                stringsAsFactors = FALSE
            )

            cipr_ref_df <<- ref_avg
            cipr_ref_annot <<- ref_annot
            """
        )
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        assert qkey is not None

        self.q = q
        self.qkey = qkey

        # pass q to R
        self._to_r(q, "q")
        ro.globalenv["q_clusters"] = obs_to_r_factor(q, qkey)

        # run CIPR
        self._t_annot_start()
        with suppress_r_console():
            ro.r(
                """
                q$cluster <- q_clusters
                Idents(q) <- "cluster"

                # DE table has the required 'gene/logfc/cluster' columns for CIPR's logfc_* modes
                allmarkers <- FindAllMarkers(q)

                CIPR(
                    input_dat = allmarkers,
                    comp_method = "logfc_dot_product",
                    reference = "custom",
                    custom_reference = cipr_ref_df,
                    custom_ref_annot = cipr_ref_annot,
                    keep_top_var = 100,
                    plot_ind = FALSE,
                    plot_top = FALSE,
                    global_results_obj = TRUE,
                    global_plot_obj = FALSE
                )

                # CIPR_top_results columns include: cluster, reference_id, identity_score, ...
                best <- CIPR_top_results %>%
                    group_by(cluster) %>%
                    slice_max(order_by = identity_score, n = 1, with_ties = FALSE)

                # map cluster -> predicted label (reference_id == ref label strings, given ref_annot)
                map <- setNames(as.character(best$reference_id), as.character(best$cluster))

                preds <- map[as.character(q$cluster)]
                """
            )
        self._t_annot_end()

        preds = list(ro.globalenv["preds"])

        # return annotations
        return Annot(preds, self.ref, self.rkey)

    def _to_r(self, adata: AnnData, name: str):

        ro.globalenv["counts"] = adata_to_r_dgCMatrix(adata, counts=True, hvg=False)

        with suppress_r_console():
            ro.r(
                f"""
                {name} <- CreateSeuratObject(counts = counts)
                {name} <- NormalizeData({name}, verbose = FALSE)
                {name} <- FindVariableFeatures({name}, selection.method = "vst", nfeatures = 2000)
                """
            )


""" --------------- SingleR ---------------"""


class SingleR(BenchModel):
    def __init__(self) -> None:
        self.id_ = "SingleR"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:
        self.ref = ref
        self.rkey = rkey

        # pass ref to R
        ro.globalenv["ref_norm"] = adata_to_r_dgCMatrix(ref, counts=False, hvg=True)
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)
        ro.r("sce_ref <- SingleCellExperiment(assays = list(logcounts = ref_norm))")

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey

        # pass q to R
        ro.globalenv["q_norm"] = adata_to_r_dgCMatrix(q, counts=False, hvg=True)
        ro.r("sce_q <- SingleCellExperiment(assays = list(logcounts = q_norm))")

        # run SingleR
        self._t_annot_start()
        ro.r(
            """
            pred <- SingleR(
                test   = sce_q,
                ref    = sce_ref,
                labels = ref_labels
            )

            # cell-level labels:
            cell_pred <- pred$labels
            names(cell_pred) <- rownames(pred)   
            """
        )
        self._t_annot_end()

        out = ro.globalenv["cell_pred"]
        labels, cell_names = list(out), list(out.names)

        # align to q.obs_names
        s = pd.Series(labels, index=cell_names)
        preds = s.loc[q.obs_names].tolist()

        # return annotations
        return Annot(preds, self.ref, self.rkey)


class SingleRcluster(BenchModel):
    def __init__(self) -> None:
        self.id_ = "SingleRcluster"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # pass ref to R
        ro.globalenv["ref_norm"] = adata_to_r_dgCMatrix(ref, counts=False, hvg=True)
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)
        ro.r("sce_ref <- SingleCellExperiment(assays = list(logcounts = ref_norm))")

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        assert qkey is not None

        self.q = q
        self.qkey = qkey

        # pass q to R
        ro.globalenv["q_norm"] = adata_to_r_dgCMatrix(q, counts=False, hvg=True)
        ro.globalenv["q_clusters"] = obs_to_r_factor(q, qkey)
        ro.r("sce_q <- SingleCellExperiment(assays = list(logcounts = q_norm))")

        # run SingleR
        self._t_annot_start()
        ro.r(
            """
            pred <- SingleR(
                test     = sce_q,
                ref      = sce_ref,
                labels   = ref_labels,
                clusters = q_clusters
            )

            # cluster-level labels:
            cluster_pred <- pred$labels
            names(cluster_pred) <- rownames(pred)
            """
        )
        self._t_annot_end()

        out = ro.globalenv["cluster_pred"]
        labels, clusters = list(out), list(out.names)

        c2l = dict(zip(clusters, labels))
        preds = q.obs[qkey].map(c2l).tolist()

        # return annotations
        return Annot(preds, self.ref, self.rkey)


""" --------------- scmap ---------------"""


class SCMAPCell(BenchModel):
    def __init__(self) -> None:
        self.id_ = "scmapcell"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:
        self.ref = ref
        self.rkey = rkey

        ro.globalenv["ref_counts"] = adata_to_r_dgCMatrix(ref, counts=True, hvg=False)
        ro.globalenv["ref_norm"] = adata_to_r_dgCMatrix(ref, counts=False, hvg=False)
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)

        self._t_ref_start()
        with suppress_r_console():
            ro.r(
                """
                set.seed(42)
                sce_ref <- SingleCellExperiment(assays = list(counts = ref_counts, logcounts = ref_norm))
                
                rowData(sce_ref)$feature_symbol <- rownames(sce_ref)
                colData(sce_ref)$label <- as.factor(unname(ref_labels))
                
                sce_ref <- selectFeatures(sce_ref, n_features = 500)
                sce_ref <- indexCell(sce_ref)
                
                # store components separately to avoid rpy2 metadata persistence issues
                scmap_subcentroids <- metadata(sce_ref)$scmap_cell_index$subcentroids
                scmap_subclusters <- metadata(sce_ref)$scmap_cell_index$subclusters
                scmap_feats <- rownames(sce_ref)[rowData(sce_ref)$scmap_features]
                """
            )
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey

        kwargs = kwargs.get("scmap", {})

        w_cell = int(kwargs.get("w_cell", 10))  # defaults
        w_agree = int(kwargs.get("w_agree", 3))
        threshold = kwargs.get("threshold", 0.5)

        assert w_agree <= w_cell

        # pass q to R
        ro.globalenv["q_counts"] = adata_to_r_dgCMatrix(q, counts=True, hvg=False)
        ro.globalenv["q_norm"] = adata_to_r_dgCMatrix(q, counts=False, hvg=False)

        # run scmap
        self._t_annot_start()
        with suppress_r_console():
            ro.r(
                f"""
                sce_q <- SingleCellExperiment(assays = list(counts = q_counts, logcounts = q_norm))
                rowData(sce_q)$feature_symbol <- rownames(sce_q)
                sce_q <- setFeatures(sce_q, scmap_feats)
                
                scmap_index <- list(subcentroids = scmap_subcentroids, subclusters = scmap_subclusters)
                res <- scmapCell(projection = sce_q, index_list = list(ref = scmap_index), w = {w_cell})
                cluster_res <- scmapCell2Cluster(res, cluster_list = list(ref = ref_labels), w = {w_agree}, threshold = {float(threshold)})
                labs <- cluster_res$combined_labs
                """
            )
        self._t_annot_end()

        preds = np.array(list(ro.globalenv["labs"]))
        preds[preds == "unassigned"] = "novel"

        # return annotations
        return Annot(preds, self.ref, self.rkey)


class SCMAPCluster(BenchModel):
    def __init__(self) -> None:
        self.id_ = "scmapcluster"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # pass ref to R -- perform initial scmap computations
        ro.globalenv["ref_counts"] = adata_to_r_dgCMatrix(ref, counts=True, hvg=False)
        ro.globalenv["ref_norm"] = adata_to_r_dgCMatrix(ref, counts=False, hvg=False)
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)

        self._t_ref_start()
        ro.r(
            """
            sce_ref <- SingleCellExperiment(assays = list(counts = ref_counts, logcounts = ref_norm))
            
            rowData(sce_ref)$feature_symbol <- rownames(sce_ref)
            colData(sce_ref)$label <- ref_labels
            
            sce_ref <- selectFeatures(sce_ref, n_features = 500)
            sce_ref <- indexCluster(sce_ref, cluster_col = "label")
            
            feats <- rownames(sce_ref)[rowData(sce_ref)$scmap_features]
            idx <- metadata(sce_ref)$scmap_cluster_index
            """
        )
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        assert qkey is not None

        self.q = q
        self.qkey = qkey

        kwargs = kwargs.get("scmap", {})
        threshold = kwargs.get("threshold", None)
        threshold = "" if threshold is None else f", threshold = {threshold}"

        # pass q to R
        ro.globalenv["q_counts"] = adata_to_r_dgCMatrix(q, counts=True, hvg=False)
        ro.globalenv["q_norm"] = adata_to_r_dgCMatrix(q, counts=False, hvg=False)
        ro.globalenv["q_clusters"] = obs_to_r_factor(q, qkey)

        # run scmap
        self._t_annot_start()
        ro.r(
            f"""
            sce_q   <- SingleCellExperiment(assays = list(counts = q_counts, logcounts = q_norm))
            rowData(sce_q)$feature_symbol   <- rownames(sce_q)
            sce_q <- setFeatures(sce_q, feats)
            
            res <- scmapCluster(projection = sce_q, index_list = list(ref = idx){threshold})
            labs <- res$scmap_cluster_labs[, "ref"]
            """
        )
        self._t_annot_end()

        preds = np.array(list(ro.globalenv["labs"]))
        preds[preds == "unassigned"] = "novel"

        # return annotations
        return Annot(preds, self.ref, self.rkey)


""" --------------- Clustifyr ---------------"""


class Clustifyr(BenchModel):
    def __init__(self) -> None:
        self.id_ = "clustifyr"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # pass ref to R -- perform initial clustifyr computations
        ro.globalenv["ref_norm"] = adata_to_r_dgCMatrix(ref, counts=False, hvg=False)
        ro.globalenv["ref_labels"] = obs_to_r_namedstrvec(ref, rkey)

        self._t_ref_start()
        with suppress_r_console():
            ro.r(
                """
                ref_meta <- data.frame(cluster = as.character(ref_labels))
                ref_avg <- average_clusters(mat = ref_norm, metadata = ref_meta, cluster_col = "cluster", if_log = TRUE)
                """
            )
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        assert qkey is not None

        self.q = q
        self.qkey = qkey

        # pass q to R
        ro.globalenv["q_norm"] = adata_to_r_dgCMatrix(q, counts=False, hvg=False)
        ro.globalenv["q_clusters"] = obs_to_r_factor(q, qkey)

        # run clustifyr
        self._t_annot_start()
        with suppress_r_console():
            ro.r(
                """
                q_meta   <- data.frame(cluster = as.character(q_clusters))
                out <- clustify(
                    input = q_norm,
                    ref_mat = ref_avg,
                    metadata = q_meta,
                    cluster_col = "cluster",
                    vec_out = TRUE,
                    obj_out = FALSE,
                    verbose = FALSE
                )
                """
            )
        self._t_annot_end()

        preds = list(ro.globalenv["out"])

        # return annotations
        return Annot(preds, self.ref, self.rkey)


""" --------------- RefCM ---------------"""


class RefCM(BenchModel):
    def __init__(self) -> None:
        self.id_ = "RefCM"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        assert qkey is not None

        kwargs = kwargs.get("RefCM", {})

        self.q = q
        self.qkey = qkey

        # run RefCM
        self._t_annot_start()
        self.model = _RCM(**kwargs)
        self.model.setref(self.ref, self.rkey)
        self.model.annotate(self.q, qkey)
        self._t_annot_end()

        # retrieve preds and cleanup
        preds = list(self.q.obs["refcm"])
        self.q.obs.drop(columns=["refcm"], inplace=True)

        # return annotations
        return Annot(preds, self.ref, self.rkey)


""" --------------- CellTypist ---------------"""


class CellTypist(BenchModel):
    def __init__(self) -> None:
        self.id_ = "CellTypist"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # train on reference
        self.gs = self.ref.var_names[self.ref.var["highly_variable"]].to_list()

        self._t_ref_start()
        with suppress_all_console():
            self.model = celltypist.train(
                self.ref[:, self.gs],
                self.rkey,
                check_expression=False,
                n_jobs=-1,
                max_iter=100,
            )
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey

        # check if genes match; retrain if not
        gs = np.intersect1d(q.var_names, self.gs)
        if len(gs) != len(self.gs):
            self._t_ref_start()
            model = celltypist.train(
                self.ref[:, gs],
                self.rkey,
                check_expression=False,
                n_jobs=-1,
                max_iter=100,
            )
            self._t_ref_end()

        else:
            gs = self.gs
            model = self.model

        # run celltypist
        self._t_annot_start()
        with suppress_all_console():
            preds = celltypist.annotate(q[:, gs], model)
        self._t_annot_end()

        preds = preds.predicted_labels["predicted_labels"].tolist()

        # return annotations
        return Annot(preds, self.ref, self.rkey)


""" --------------- SVM ---------------"""


class SVM(BenchModel):
    def __init__(self) -> None:
        self.id_ = "SVM"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey

        # train on reference
        self.gs = self.ref.var_names[self.ref.var["highly_variable"]].to_list()
        self._t_ref_start()
        self.model = LinearSVC()
        self.model.fit(self.ref[:, self.gs].X, self.ref.obs[self.rkey])
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey

        # check if genes match; retrain if not
        gs = np.intersect1d(q.var_names, self.gs)
        if len(gs) != len(self.gs):
            self._t_ref_start()
            model = LinearSVC()
            model.fit(self.ref[:, gs].X, self.ref.obs[self.rkey])
            self._t_ref_end()
        else:
            gs = self.gs
            model = self.model

        # run SVM
        self._t_annot_start()
        preds = model.predict(self.q[:, gs].X)
        self._t_annot_end()

        # return annotations
        return Annot(preds, self.ref, self.rkey)


""" --------------- SCANVI ---------------"""


class SCANVI(BenchModel):
    def __init__(self) -> None:
        self.id_ = "scANVI"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:

        self.ref = ref
        self.rkey = rkey
        batch_key = kwargs.get("batch_key", None)

        self.gs = self.ref.var_names[self.ref.var["highly_variable"]].to_list()

        self.ref = self.ref[:, self.gs].copy()
        self.ref.X = self.ref.layers["counts"]
        self.ref.layers.clear()

        self._t_ref_start()
        scvi.model.SCVI.setup_anndata(
            self.ref, layer=None, batch_key=batch_key, labels_key=self.rkey
        )

        self.scvi_ref = scvi.model.SCVI(
            self.ref,
            use_layer_norm="both",
            use_batch_norm="none",
            encode_covariates=True,
            dropout_rate=0.2,
            n_layers=2,
        )
        self.scvi_ref.train()

        self.scanvi_ref = scvi.model.SCANVI.from_scvi_model(
            self.scvi_ref,
            unlabeled_category="Unknown",
            labels_key=self.rkey,
        )
        self.scanvi_ref.train(max_epochs=20, n_samples_per_label=100)
        self._t_ref_end()

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey

        gs = np.intersect1d(q.var_names, self.gs)
        self.q = self.q[:, gs].copy()
        self.q.X = self.q.layers["counts"]
        self.q.layers.clear()
        self.q.obs.drop(columns=[self.rkey], inplace=True, errors="ignore")

        # run SCANVI
        self._t_annot_start()
        scvi.model.SCANVI.prepare_query_anndata(self.q, self.scanvi_ref)
        self.scanvi_q = scvi.model.SCANVI.load_query_data(self.q, self.scanvi_ref)

        self.scanvi_q.train(
            max_epochs=100,
            plan_kwargs={"weight_decay": 0.0},
            check_val_every_n_epoch=10,
        )
        preds = self.scanvi_q.predict()
        self._t_annot_end()

        return Annot(preds, self.ref, self.rkey)


""" --------------- SCALEX ---------------"""


class SCALEX(BenchModel):
    def __init__(self) -> None:
        self.id_ = "SCALEX"
        super().__init__()

    def setref(self, ref: AnnData, rkey: str, **kwargs) -> None:
        self.ref = ref
        self.rkey = rkey

        kwargs = kwargs.get("SCALEX", {})
        self.rid = kwargs.get("rid", "r")
        self.fpath = kwargs.get("fpath", "SCALEX")
        batch_name = kwargs.get("batch_name", "batch")
        batch_name = "batch" if batch_name is None else batch_name
        os.makedirs(self.fpath, exist_ok=True)

        # run scalex if not pre-run on reference
        if self.rid == "r" or not os.path.isfile(f"{self.fpath}/{self.rid}/adata.h5ad"):
            rnorm, self.ref.X = self.ref.X, self.ref.layers["counts"]

            print(kwargs)
            print(batch_name)
            self._t_ref_start()
            _SCALEX(
                self.ref,
                batch_name=batch_name,
                min_features=0,
                min_cells=0,
                outdir=f"{self.fpath}/{self.rid}/",
                ignore_umap=True,
            )
            self._t_ref_end()

            self.ref.X = rnorm

    def annotate(self, q: AnnData, qkey: str | None = None, **kwargs) -> list:

        self.q = q
        self.qkey = qkey
        kwargs = kwargs.get("SCALEX", {})
        self.qid = kwargs.get("qid", "q")
        batch_name = kwargs.get("batch_name", "batch")
        batch_name = "batch" if batch_name is None else batch_name

        # check if projection had already been done
        self._t_annot_start()
        if self.qid == "q" or not os.path.isfile(
            f"{self.fpath}/{self.qid}_{self.rid}/adata.h5ad"
        ):
            qnorm, self.q.X = self.q.X, self.q.layers["counts"]
            sclx = _SCALEX(
                self.q,
                batch_name=batch_name,
                min_features=0,
                min_cells=0,
                outdir=f"{self.fpath}/{self.qid}_{self.rid}/",
                projection=f"{self.fpath}/{self.rid}/",
                ignore_umap=True,
            )
            self.q.X = qnorm
        else:
            sclx = sc.read_h5ad(f"{self.fpath}/{self.qid}_{self.rid}/adata.h5ad")

        # complete label transfer
        sclx_q = sclx[sclx.obs.projection == "query"]
        sclx_ref = sclx[sclx.obs.projection == "reference"]

        preds = label_transfer(sclx_ref, sclx_q, rep="latent", label=self.rkey)
        self._t_annot_end()

        return Annot(preds, self.ref, self.rkey)
