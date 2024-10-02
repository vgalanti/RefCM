import sys

sys.path.append("../src/")

import os
import json
import time
import torch
import config
import pandas as pd
import numpy as np
import scanpy as sc
import logging
import seaborn as sns

# import rpy2.robjects as ro
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# import rpy2.robjects as ro


from typing import List, Dict, Tuple, TypedDict, TypeAlias, NamedTuple
from anndata import AnnData
from itertools import product
from collections import defaultdict
from scipy.sparse import issparse

# from rpy2.robjects import pandas2ri
from plotly.subplots import make_subplots


# benchmark model imports
import scvi
import celltypist

from refcm import RefCM
from sklearn.svm import LinearSVC

# config
# pandas2ri.activate()
scvi.settings.seed = 0
log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


# # NOTE ensure that the R environment is
# # correctly setup with necessary library imports
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages('Seurat')


class BenchModel:
    """
    A common interface for all benchmarked methods, as
    reference mapping and reference cluster mapping algorithms.
    """

    rm_id: str
    rcm_id: str

    def __init__(self) -> None:
        pass

    def setref(self, ref: AnnData, ref_key: str) -> None:
        """
        Set the reference dataset reference dataset.

        Parameters
        ----------
        ref: AnnData
            The reference dataset with raw counts in .X
        ref_key: str
            .obs key for reference labels.
        """

        self.ref = ref
        self.ref_key = ref_key

    def annotate(self, q: AnnData, q_key: str, q_mclusters: str | None = None) -> None:
        """
        Annotates the query dataset using the reference data, and adds
        its predictions as a new column (self.id) under the query's .obs.

        Parameters
        ----------
        q: AnnData
            The query dataset with raw counts in .X
        q_key: str
            The query .obs key with the true labels; used for
            clustering information for majority voting.
        q_mclusters:
            The query .obs key with manual clusters; used to
            eval RCM methods as RM methods
        """
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

    def _mv(self) -> None:
        """
        Apply majority voting based on previous predictions and
        ground truth clustering.

        Assumes .annotate() has already been run()
        """
        self.q.obs[self.rcm_id] = self.q.obs[self.rm_id]
        truth_labels = sorted(self.q.obs[self.q_key].unique().tolist())
        for cluster in truth_labels:
            cmask = self.q.obs[self.q_key] == cluster
            mv = self.q.obs.loc[cmask, self.rm_id].value_counts().idxmax()
            self.q.obs.loc[cmask, self.rcm_id] = mv

    def eval_(self, as_rcm: bool = True) -> Tuple:
        """
        Evaluates a model on a query/reference pair.

        Parameters
        ----------
        as_rcm: bool = True
            Whether to return the performance of the model as RCM method
            else as RM method.

        Returns
        -------
        acc: float
            The % accuracy.
        cfmatrix: List[List[int]]
            The confusion matrix [true query labels] x [reference labels + novel]
        q_labels: List[str]
            The query labels (row/y axis in confusion matrix)
        ref_labels: List[str]
            Reference labels + novel (column/x axis in confusion matrix)
        """
        pred_key = self.rcm_id if as_rcm else self.rm_id

        q_labels = sorted(self.q.obs[self.q_key].unique().tolist())
        ref_labels = sorted(self.ref.obs[self.ref_key].unique().tolist()) + ["novel"]

        novel_labels = list(set(q_labels) - set(ref_labels))

        correct = 0
        cfmatrix = np.zeros((len(q_labels), len(ref_labels)))

        for i in range(len(q_labels)):
            true_mask = self.q.obs[self.q_key] == q_labels[i]

            for j in range(len(ref_labels)):
                pred_mask = self.q.obs[pred_key] == ref_labels[j]

                if (q_labels[i] == ref_labels[j]) or (
                    q_labels[i] in novel_labels and ref_labels[j] == "novel"
                ):
                    correct += len(self.q.obs[(true_mask & pred_mask)])

                cfmatrix[i, j] = len(self.q.obs[(true_mask & pred_mask)])

        acc = correct / self.q.X.shape[0]
        return acc, cfmatrix.astype(int).tolist(), q_labels, ref_labels

    # TODO for add option to normalize by row/column
    def plot_cfmatrix(
        self,
        cfmatrix: List[List[int]],
        q_labels: List[str],
        ref_labels: List[str],
        width=750,
        height=750,
        show_nums: bool = False,
        angle_ticks: bool = True,
    ) -> None:
        """
        Plots the confusion matrix resulting from the above function.
        """
        fig = px.imshow(
            cfmatrix, x=ref_labels, y=q_labels, color_continuous_scale="Blues"
        )

        if show_nums:
            for i in range(len(q_labels)):
                for j in range(len(ref_labels)):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{cfmatrix[i][j]}",
                        showarrow=False,
                        # bgcolor="white",
                        # opacity=0.2,
                    )
            fig.update_annotations(font=dict(color="white"))

        fig.update_layout(width=width, height=height)
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)

        fig.update_xaxes(tickangle=-90)
        if angle_ticks:
            fig.update_xaxes(tickangle=-45)

        fig.show()


class RCM(BenchModel):
    """
    RefCM wrapper.
    """

    def __init__(self) -> None:
        self.rm_id = "LD-RefCM"
        self.rcm_id = "RefCM"
        super().__init__()

    def annotate(
        self, q: AnnData, q_key: str, q_mclusters: str | None = None, **kwargs
    ) -> None:
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

        # we are given a manual clustering key
        # to evaluate as a reference mapping algorithm
        if self.q_mclusters is not None:

            t_start = time.perf_counter()
            self.model = RefCM(**kwargs)
            self.model.setref(self.ref, "ref", self.ref_key)
            self.model.annotate(self.q, "q", q_mclusters)
            t_elapsed = time.perf_counter() - t_start

            log.debug(f"[*] {self.rm_id} completed: {t_elapsed/60:.1f}mins")
            self.q.obs.rename(columns={"refcm_annot": self.rm_id}, inplace=True)

        # RefCM run on true clusters
        t_start = time.perf_counter()
        self.model = RefCM(**kwargs)
        self.model.setref(self.ref, "ref", self.ref_key)
        self.model.annotate(self.q, "q", q_key)
        t_elapsed = time.perf_counter() - t_start

        log.debug(f"[*] {self.rcm_id} completed: {t_elapsed/60:.1f}mins")
        self.q.obs.rename(columns={"refcm_annot": self.rcm_id}, inplace=True)


class CellTypist(BenchModel):

    def __init__(self) -> None:
        self.rm_id = "Celltypist"
        self.rcm_id = "MV-Celltypist"
        super().__init__()

    def annotate(self, q: AnnData, q_key: str, q_mclusters: str | None = None) -> None:
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

        # sum & log1p normalization as per official documentation
        q_sums = self.q.X.sum(axis=1).reshape((-1, 1))
        ref_sums = self.ref.X.sum(axis=1).reshape((-1, 1))

        self.q.X = np.log1p(1e4 * self.q.X / q_sums)
        self.ref.X = np.log1p(1e4 * self.ref.X / ref_sums)

        if issparse(self.q.X):
            self.q.X = self.q.X.toarray()

        if issparse(self.ref.X):
            self.ref.X = self.ref.X.toarray()

        # skip downsampling since dataset sizes are <1e6
        # use all intersecting genes for higher accuracy
        gs = np.intersect1d(self.q.var_names, self.ref.var_names)

        # model training
        t_start = time.perf_counter()
        model = celltypist.train(
            self.ref[:, gs],
            self.ref_key,
            check_expression=False,
            n_jobs=10,
            max_iter=100,
        )
        t_elapsed = time.perf_counter() - t_start
        log.debug(f"[*] {self.rm_id} completed: {t_elapsed/60:.1f}mins")

        # model prediction
        preds = celltypist.annotate(q[:, gs], model)
        self.q.obs[self.rm_id] = preds.predicted_labels["predicted_labels"]

        # cleanup so datasets remain unchanged between executions
        self.q.X = np.expm1(self.q.X) * q_sums / 1e4
        self.ref.X = np.expm1(self.ref.X) * ref_sums / 1e4

        # apply majority voting
        self._mv()


class SCANVI(BenchModel):

    def __init__(self) -> None:
        self.rm_id = "SCANVI"
        self.rcm_id = "MV-SCANVI"
        super().__init__()

    def annotate(self, q: AnnData, q_key: str, q_mclusters: str | None = None) -> None:
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

        # sum & log1p normalization as per official documentation
        q_sums = self.q.X.sum(axis=1).reshape((-1, 1))
        ref_sums = self.ref.X.sum(axis=1).reshape((-1, 1))

        self.q.X = np.log1p(1e4 * self.q.X / q_sums)
        self.ref.X = np.log1p(1e4 * self.ref.X / ref_sums)

        # TODO might need to actually use gs...
        # gs = np.intersect1d(self.q.var_names, self.ref.var_names)

        # model training
        t_start = time.perf_counter()
        scvi.model.SCVI.setup_anndata(self.ref)

        scvi_ref = scvi.model.SCVI(
            self.ref,
            use_layer_norm="both",
            use_batch_norm="none",
            encode_covariates=True,
            dropout_rate=0.2,
            n_layers=2,
        )
        scvi_ref.train()

        scanvi_ref = scvi.model.SCANVI.from_scvi_model(
            scvi_ref,
            unlabeled_category="Unknown",
            labels_key=self.ref_key,
        )

        scanvi_ref.train(max_epochs=20, n_samples_per_label=100)

        # model prediction
        scvi.model.SCANVI.prepare_query_anndata(q, scanvi_ref)

        scanvi_query = scvi.model.SCANVI.load_query_data(q, scanvi_ref)

        scanvi_query.train(
            max_epochs=100,
            plan_kwargs={"weight_decay": 0.0},
            check_val_every_n_epoch=10,
        )

        q.obs[self.rm_id] = scanvi_query.predict()

        t_elapsed = time.perf_counter() - t_start
        log.debug(f"[*] {self.rm_id} completed: {t_elapsed/60:.1f}mins")

        # cleanup so datasets remain unchanged between executions
        self.q.X = np.expm1(self.q.X) * q_sums / 1e4
        self.ref.X = np.expm1(self.ref.X) * ref_sums / 1e4

        # apply majority voting
        self._mv()


class SVM(BenchModel):

    def __init__(self) -> None:
        self.rm_id = "SVM"
        self.rcm_id = "MV-SVM"
        super().__init__()

    def annotate(self, q: AnnData, q_key: str, q_mclusters: str | None = None) -> None:
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

        # sum & log1p normalization as per official documentation
        q_sums = self.q.X.sum(axis=1).reshape((-1, 1))
        ref_sums = self.ref.X.sum(axis=1).reshape((-1, 1))

        self.q.X = np.log1p(1e4 * self.q.X / q_sums)
        self.ref.X = np.log1p(1e4 * self.ref.X / ref_sums)

        if issparse(self.q.X):
            self.q.X = self.q.X.toarray()

        if issparse(self.ref.X):
            self.ref.X = self.ref.X.toarray()

        # use all intersecting genes for higher accuracy
        gs = np.intersect1d(self.q.var_names, self.ref.var_names)
        sc.pp.highly_variable_genes(self.ref, n_top_genes=2000)
        hvg = self.ref.var["highly_variable"].index.to_list()
        gs = gs if len(np.intersect1d(gs, hvg)) < 1000 else np.intersect1d(gs, hvg)

        # model training
        t_start = time.perf_counter()
        svm = LinearSVC()
        svm.fit(self.ref[:, gs].X, self.ref.obs[self.ref_key])

        t_elapsed = time.perf_counter() - t_start
        log.debug(f"[*] {self.rm_id} completed: {t_elapsed/60:.1f}mins")

        # model prediction
        self.q.obs[self.rm_id] = svm.predict(self.q[:, gs].X)

        # cleanup so datasets remain unchanged between executions
        self.q.X = np.expm1(self.q.X) * q_sums / 1e4
        self.ref.X = np.expm1(self.ref.X) * ref_sums / 1e4

        self._mv()


class Seurat(BenchModel):

    def __init__(self) -> None:
        self.rm_id = "Seurat"
        self.rcm_id = "MV-Seurat"
        super().__init__()

    def annotate(self, q: AnnData, q_key: str, q_mclusters: str | None = None) -> None:
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

        # Pass objects to R
        log.debug("passing objects to R")
        ro.globalenv["q"] = self.q.to_df()
        ro.globalenv["ref"] = self.ref.to_df()
        ro.globalenv["q_labels"] = self.q.obs[q_key]
        ro.globalenv["ref_labels"] = self.ref.obs[self.ref_key]

        # model training
        log.debug("running R code")
        t_start = time.perf_counter()

        ro.r(
            """
            library(Seurat)
            
            # Convert data frames to Seurat objects
            q <- CreateSeuratObject(counts = t(as.matrix(q)))
            ref <- CreateSeuratObject(counts = t(as.matrix(ref)))

            # Normalize the data
            q <- NormalizeData(q, verbose = FALSE)
            ref <- NormalizeData(ref, verbose = FALSE)

            # Feature selection
            q <- FindVariableFeatures(q, selection.method = "vst", nfeatures = 2000)
            ref <- FindVariableFeatures(ref, selection.method = "vst", nfeatures = 2000)

            # Find transfer anchors
            anchors <- FindTransferAnchors(reference = ref, query = q, dims = 1:30)

            # Transfer cell type labels from reference to query
            preds <- TransferData(anchorset = anchors, refdata = ref_labels, dims = 1:30)

            # Add predicted cell types to the query Seurat object
            q <- AddMetaData(object = q, metadata = preds)

            # Extract predicted cell types and store them
            preds <- q$predicted.id
        """
        )

        self.q.obs[self.rm_id] = ro.globalenv["preds"]
        t_elapsed = time.perf_counter() - t_start
        log.debug(f"[*] {self.rm_id} completed: {t_elapsed/60:.1f}mins")

        self._mv()


class CIPR(BenchModel):

    def __init__(self) -> None:
        self.rm_id = "LD-CIPR"
        self.rcm_id = "CIPR"
        super().__init__()

    def annotate(self, q: AnnData, q_key: str, q_mclusters: str | None = None) -> None:
        self.q = q
        self.q_key = q_key
        self.q_mclusters = q_mclusters

        r_code = """
            library(Seurat)
            library(CIPR)
            
            # Convert data frames to Seurat objects
            q <- CreateSeuratObject(counts = t(as.matrix(q)))
            ref <- CreateSeuratObject(counts = t(as.matrix(ref)))

            # Normalize the data
            q <- NormalizeData(q, verbose = FALSE)
            ref <- NormalizeData(ref, verbose = FALSE)

            # # Feature selection
            # q <- FindVariableFeatures(q, selection.method = "vst", nfeatures = 2000)
            # ref <- FindVariableFeatures(ref, selection.method = "vst", nfeatures = 2000)

            # CIPR
            q <- FindAllMarkers(q)
            q <- AverageExpression(q)
            
            # avgexp <- as.data.frame(avgexp$RNA)
            # avgexp$gene <- rownames(avgexp)
            
            
            cipr_output <- CIPR(
                input_dat = q,
                clusters = q_labels,
                comp_method = "logfc_dot_product",
                reference="custom",
                custom_reference = ref,
                custom_ref_annot = ref_labels,
                keep_top_var = 100,
                global_results_obj = F,
                plot_top = F
            )
            
            preds <- cipr_output$annotated_data
            # Save the final annotations to a CSV file
            
            write.csv(preds, "final_annotations.csv", row.names = FALSE)
        """

        # ensure gene names match
        gs = np.intersect1d(self.q.var_names, self.ref.var_names)

        # pass to R
        log.debug("passing objects to R...")
        ro.globalenv["q"] = self.q[:, gs].to_df()
        ro.globalenv["ref"] = self.ref[:, gs].to_df()
        ro.globalenv["ref_labels"] = self.ref.obs[self.ref_key]

        if self.q_mclusters is not None:
            ro.globalenv["q_labels"] = self.q.obs[q_mclusters]
            # model training
            log.debug(f"{self.rm_id} R code running")
            t_start = time.perf_counter()
            ro.r(r_code)
            t_elapsed = time.perf_counter() - t_start
            print(f"{self.rm_id} completed: {t_elapsed/60:.1f}mins")

            preds = ro.globalenv["preds"]
            self.q.obs[self.rm_id] = preds

        ro.globalenv["q_labels"] = self.q.obs[q_key]
        # model training
        log.debug(f"{self.rcm_id} R code running")
        t_start = time.perf_counter()
        ro.r(r_code)
        t_elapsed = time.perf_counter() - t_start
        print(f"{self.rcm_id} completed: {t_elapsed/60:.1f}mins")

        preds = ro.globalenv["preds"]
        self.q.obs[self.rcm_id] = preds


PANCREAS = [
    "pancreas_celseq",
    "pancreas_celseq2",
    "pancreas_fluidigmc1",
    "pancreas_inDrop1",
    "pancreas_inDrop2",
    "pancreas_inDrop3",
    "pancreas_inDrop4",
    "pancreas_smarter",
    "pancreas_smartseq2",
]

ALLENBRAIN = ["ALM", "MTG", "VISp"]
MONKEY = ["mag_old", "mag_young"]
FZFISH = ["frog", "zebrafish"]

KEYS = (
    {p: "celltype" for p in PANCREAS}
    | {b: "labels34" for b in ALLENBRAIN}
    | {m: "celltype" for m in MONKEY}
    | {f: "cell_type" for f in FZFISH}  # TODO check this
)


class Benchmark(NamedTuple):
    acc: float
    cfmatrix: List[List[int]]
    q_labels: List[str]
    ref_labels: List[str]
    annotations: List[str]


# {method: {qid: {rid: Benchmark}}}
Benchmarks: TypeAlias = Dict[str, Dict[str, Dict[str, Benchmark]]]


def load_benchmarks() -> Benchmarks:
    if os.path.exists("benchmarks.json"):
        with open("benchmarks.json", "r") as f:
            benchmarks = json.load(f)
    else:
        benchmarks = {}

    for m in benchmarks:
        for qid in benchmarks[m]:
            for rid in benchmarks[m][qid]:
                benchmarks[m][qid][rid] = Benchmark(*benchmarks[m][qid][rid])

    return benchmarks


def add_benchmark(
    benchmarks: Benchmarks, m: str, qid: str, rid: str, perf: Benchmark
) -> None:
    if benchmarks.get(m) is None:
        benchmarks[m] = {}
    if benchmarks[m].get(qid) is None:
        benchmarks[m][qid] = {}
    benchmarks[m][qid][rid] = perf


def save_benchmarks(benchmarks: Benchmarks) -> None:
    with open("benchmarks.json", "w") as f:
        json.dump(benchmarks, f)


# Results needed for fig 2a
def benchmark_pancreas() -> None:
    benchmarks: Benchmarks = load_benchmarks()

    for qid, rid in product(PANCREAS, PANCREAS):
        q = sc.read_h5ad(f"../data/{qid}.h5ad")
        ref = sc.read_h5ad(f"../data/{rid}.h5ad")

        # TODO remove this, for scanvi
        if set(q.obs.celltype.unique()) != set(ref.obs.celltype.unique()):
            continue

        # RefCM
        # m = RCM()
        # m.setref(ref, KEYS[rid])
        # m.annotate(q, KEYS[qid], None, discovery_threshold=0, pdist="euclidean")

        # perf = Benchmark(*m.eval_(), q.obs[m.rcm_id].tolist())
        # add_benchmark(benchmarks, m.rcm_id, qid, rid, perf)
        print(f"{qid} | {rid}")
        # benchmarked models
        models = [SCANVI]  # Seurat, SVM]  # SCANVI TODO [CIPR, ClustifyR, ]
        for model in models:
            m = model()
            m.setref(ref, KEYS[rid])
            m.annotate(q, KEYS[qid])

            perf = Benchmark(*m.eval_(), q.obs[m.rcm_id].tolist())
            add_benchmark(benchmarks, m.rcm_id, qid, rid, perf)

    # save_benchmarks(benchmarks)
    return benchmarks


# Results needed for fig2b
# assumes manual leiden clustering has already been done
def benchmark_brain_monkey() -> None:
    benchmarks: Benchmarks = load_benchmarks()

    with open("leiden.json") as f:
        leiden_clusters = json.load(f)

    cs = list(product(ALLENBRAIN, ALLENBRAIN)) + list(product(MONKEY, MONKEY))
    cs = [(q, r) for q, r in cs if q != r]

    for qid, rid in cs:
        q = sc.read_h5ad(f"../data/{qid}.h5ad")
        ref = sc.read_h5ad(f"../data/{rid}.h5ad")

        # add manual leiden clusters
        q.obs["leiden"] = leiden_clusters[qid]
        q.obs.leiden = q.obs.leiden.astype(str)

        # RefCM
        # m = RCM()
        # m.setref(ref, KEYS[rid])
        # m.annotate(q, KEYS[qid], "leiden", discovery_threshold=0)

        # perf = Benchmark(*m.eval_(True), q.obs[m.rcm_id].tolist())
        # add_benchmark(benchmarks, m.rcm_id, qid, rid, perf)

        # perf = Benchmark(*m.eval_(False), q.obs[m.rm_id].tolist())
        # add_benchmark(benchmarks, m.rm_id, qid, rid, perf)

        # benchmarked models
        models = [SCANVI]  # CellTypist, SVM, Seurat]  # SCANVI TODO [CIPR, ClustifyR, ]
        for model in models:
            m = model()
            m.setref(ref, KEYS[rid])
            m.annotate(q, KEYS[qid], "leiden")

            perf = Benchmark(*m.eval_(True), q.obs[m.rcm_id].tolist())
            add_benchmark(benchmarks, m.rcm_id, qid, rid, perf)

            perf = Benchmark(*m.eval_(False), q.obs[m.rm_id].tolist())
            add_benchmark(benchmarks, m.rm_id, qid, rid, perf)

        save_benchmarks(benchmarks)


# Results needed for fig 2c
def benchmark_fzfish() -> None:
    benchmarks: Benchmarks = load_benchmarks()

    for qid, rid in [("frog", "zebrafish"), ("zebrafish", "frog")]:
        q = sc.read_h5ad(f"../data/{qid}.h5ad")
        ref = sc.read_h5ad(f"../data/{rid}.h5ad")

        # RefCM
        # m = RCM()
        # m.setref(ref, KEYS[rid])
        # m.annotate(q, KEYS[qid])

        # perf = Benchmark(*m.eval_(), q.obs[m.rcm_id].tolist())
        # add_benchmark(benchmarks, m.rcm_id, qid, rid, perf)

        # benchmarked models
        models = [CellTypist]  # , Seurat, SVM]  # SCANVI TODO [CIPR, ClustifyR, ]
        for model in models:
            m = model()
            m.setref(ref, KEYS[rid])
            m.annotate(q, KEYS[qid])

            perf = Benchmark(*m.eval_(), q.obs[m.rcm_id].tolist())
            add_benchmark(benchmarks, m.rcm_id, qid, rid, perf)

    save_benchmarks(benchmarks)


def plot_pancreas_perf(
    model: str, write: bool = False, width: int = 750, height: int = 750
):
    benchmarks: Benchmarks = load_benchmarks()

    if benchmarks.get(model) is None:
        log.error(f"{model} does not have any saved benchmarks")
        return

    benchmarks = benchmarks.get(model)

    accs = np.zeros((len(PANCREAS), len(PANCREAS)))

    for i, qid in enumerate(PANCREAS):
        b = benchmarks.get(qid)
        if b is None:
            log.error(f"{model} missing pancreas benchmarks")
            return

        for j, rid in enumerate(PANCREAS):
            perf = b.get(rid)
            if perf is None:
                log.error(f"{model} missing pancreas benchmarks")
                return

            accs[i][j] = perf.acc

    fig = px.imshow(
        accs,
        color_continuous_scale="Blues",
        x=PANCREAS,
        y=PANCREAS,
        zmin=0,
        zmax=1,
        title=f"{model}",
    )

    fig.update_layout(width=width, height=height)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(tickangle=-90)

    if write:
        os.makedirs("fig2", exist_ok=True)
        fig.write_image(f"fig2/pancreas_{model}.png")

    return fig


def plot_brain_monkey_perf(
    models: List[str],
    brain: bool = True,
    monkey: bool = True,
    write: bool = False,
    width: int = 750,
    height: int = 1000,
):
    benchmarks: Benchmarks = load_benchmarks()

    cs = []
    if brain:
        cs += list(product(ALLENBRAIN, ALLENBRAIN))
    if monkey:
        cs += list(product(MONKEY, MONKEY))

    cs = [(q, r) for q, r in cs if q != r]

    accs = np.zeros((len(models), len(cs)))
    for i, m in enumerate(models):
        for j, (q, r) in enumerate(cs):
            b = benchmarks.get(m)

            if b is None:
                log.error(f"{m} does not have any saved benchmarks")
                return

            b = b.get(q)
            if b is None:
                log.error(f"{m} missing {q} | {r} benchmarks")
                return

            b = b.get(r)
            if b is None:
                log.error(f"{m} missing {q} | {r} benchmarks")
                return

            accs[i][j] = b.acc

    accs = accs.round(2)

    x = [f"{q} | {r}" for q, r in cs]
    y = models

    fig = px.imshow(
        accs, color_continuous_scale="Blues", x=x, y=y, zmin=0, zmax=1, text_auto=True
    )

    fig.update_layout(width=width, height=height)
    fig.update_xaxes(dtick=1, side="top", tickangle=-60)
    fig.update_yaxes(dtick=1, scaleanchor="x", scaleratio=1)

    if write:
        os.makedirs("fig2", exist_ok=True)
        fig.write_image(f"fig2/brain_monkey.png", scale=2)

    return fig


def plot_fzfish_perf(
    models: List[str],
    tasks=[("frog", "zebrafish"), ("zebrafish", "frog")],
    write: bool = False,
    width: int = 1200,
    height: int = 400,
):
    benchmarks: Benchmarks = load_benchmarks()

    data = []
    for m in models:
        b = benchmarks.get(m)
        if b is None:
            print(f"{m} does not have any benchmarks available")
            break

        for q, r in tasks:
            perf = b.get(q)
            if perf is None:
                print(f"{m} does not have {q}  |  {r} benchmark")
                break

            perf = perf.get(r)
            if perf is None:
                print(f"{m} does not have {q}  |  {r} benchmark")
                break

            data.append([m, f"{q} | {r}", round(perf.acc, 2)])

    df = pd.DataFrame(data, columns=["method", "task", "accuracy"])
    fig = px.bar(
        df,
        x="task",
        y="accuracy",
        color="method",
        barmode="group",
        # text="accuracy",
        range_y=[0, 1.1],
    )
    fig.update_layout(width=width, height=height, plot_bgcolor="white", bargap=0.7)
    fig.update_traces(textposition="outside", textfont_size=16)

    fig.update_xaxes(showgrid=False, showline=True, linecolor="black", title=None)
    fig.update_yaxes(showline=True, linecolor="black", showgrid=True, title="accuracy")

    if write:
        os.makedirs("fig2", exist_ok=True)
        fig.write_image(f"fig2/fzfish.png")

    return fig


def plot_umap(
    ds_id: str, write: bool = False, width: int = 750, height: int = 750, msize: int = 3
):

    # load the dataset for the umap
    ds = sc.read_h5ad(f"../data/{ds_id}.h5ad")

    # preprocess for umap
    sc.pp.normalize_total(ds, target_sum=1e4)
    sc.pp.log1p(ds)
    sc.tl.pca(ds)
    sc.pp.neighbors(ds)
    sc.tl.umap(ds)

    # compute information to plot
    df = pd.DataFrame(
        {
            "x": ds.obsm["X_umap"][:, 0],
            "y": ds.obsm["X_umap"][:, 1],
            "color": ds.obs[KEYS[ds_id]],
        }
    ).sort_values("color")

    # plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="color",
        color_discrete_sequence=sc.plotting.palettes.default_102,
    )

    fig.update_layout(width=width, height=height, plot_bgcolor="white")
    fig.update_traces(marker=dict(size=msize))
    fig.update_legends(
        itemsizing="constant",
        title=None,
        orientation="v",
        xanchor="left",
        yanchor="top",
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        mirror=True,
        linecolor="black",
        showticklabels=False,
        title=None,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=True,
        mirror=True,
        linecolor="black",
        showgrid=False,
        showticklabels=False,
        title=None,
    )

    if write:
        os.makedirs("fig3", exist_ok=True)
        fig.write_image(f"fig3/umap_{ds_id}.png")

    return fig


def plot_accuracy_umap(
    model: str,
    qid: str,
    rid: str,
    write: bool = False,
    width: int = 750,
    height: int = 750,
    msize: int = 3,
):

    benchmarks = load_benchmarks()

    b = benchmarks.get(model)

    if b is None:
        log.error(f"{model} does not have any saved benchmarks")
        return

    b = b.get(qid)
    if b is None:
        log.error(f"{model} missing {qid} | {rid} benchmarks")
        return

    b = b.get(rid)
    if b is None:
        log.error(f"{model} missing {qid} | {rid} benchmarks")
        return

    # load the dataset for the umap
    ds = sc.read_h5ad(f"../data/{qid}.h5ad")

    # preprocess for umap
    sc.pp.normalize_total(ds, target_sum=1e4)
    sc.pp.log1p(ds)
    sc.tl.pca(ds)
    sc.pp.neighbors(ds)
    sc.tl.umap(ds)

    # compute information to display
    ds.obs[model] = b.annotations

    df = pd.DataFrame({"x": ds.obsm["X_umap"][:, 0], "y": ds.obsm["X_umap"][:, 1]})
    colors = (
        (ds.obs[KEYS[qid]] == ds.obs[model])
        .astype(int)
        .map({0: "crimson", 1: "lightgreen"})
    )
    acc = (ds.obs[KEYS[qid]] == ds.obs[model]).mean()

    # plot
    fig = px.scatter(
        df, x="x", y="y", title=f"{model} <br><sup>accuracy: {acc:.2f} </sup>"
    )
    fig.update_layout(width=width, height=height, plot_bgcolor="white")
    fig.update_traces(marker=dict(size=msize, color=colors))
    fig.update_legends(
        itemsizing="constant",
        title=None,
        orientation="v",
        xanchor="left",
        yanchor="top",
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        mirror=True,
        linecolor="black",
        showticklabels=False,
        title=None,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        showline=True,
        mirror=True,
        linecolor="black",
        showgrid=False,
        showticklabels=False,
        title=None,
    )
    if write:
        os.makedirs("fig3", exist_ok=True)
        fig.write_image(f"fig3/umap_{model}_{qid}_{rid}.png")

    return fig
