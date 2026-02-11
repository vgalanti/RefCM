import os
from benchdb import *
from benchutils import *
from benchmodels import *

import scipy.sparse as sp
import pandas as pd
import numpy as np
import shutil

print("loading tmuris...")
REAL_ADATA = load_adata("../data/tmuris_senis_droplet.h5ad")
print("loaded tmuris")

LABEL_KEY = "cell_ontology_class"


def generate_upsampled(n_cells, n_celltypes, query_frac=0.2):
    """
    Select n_celltypes, split into train/test, bootstrap to n_cells.
    """
    available_types = REAL_ADATA.obs[LABEL_KEY].unique()

    if n_celltypes >= len(available_types):
        selected_types = available_types
    else:
        selected_types = np.random.choice(
            available_types, size=n_celltypes, replace=False
        )

    subset_mask = REAL_ADATA.obs[LABEL_KEY].isin(selected_types).values
    subset_indices = np.where(subset_mask)[0]
    np.random.shuffle(subset_indices)

    n_real_subset = len(subset_indices)
    n_q_real = int(n_real_subset * query_frac)

    real_q_pool = subset_indices[:n_q_real]
    real_ref_pool = subset_indices[n_q_real:]

    target_q = int(n_cells * query_frac)
    target_ref = n_cells - target_q

    q_boot_idx = np.random.choice(real_q_pool, size=target_q, replace=True)
    ref_boot_idx = np.random.choice(real_ref_pool, size=target_ref, replace=True)

    q = REAL_ADATA[q_boot_idx].copy()
    ref = REAL_ADATA[ref_boot_idx].copy()

    q.obs_names = [f"Q_{i}" for i in range(target_q)]
    ref.obs_names = [f"R_{i}" for i in range(target_ref)]

    if "counts" in REAL_ADATA.layers:
        q.X = q.layers["counts"].copy()
        ref.X = ref.layers["counts"].copy()

    return q, ref


models = [
    RefCM,
    Clustifyr,
    CIPR,
    SingleRcluster,
    CellTypist,
    SVM,
    Seurat,
    SingleR,
    SCMAPCell,
    SCMAPCluster,
]
key = "cell_ontology_class"
kwargs = {
    "scmapcell": {"w_agree": 1, "threshold": 0.0},
    "scmapcluster": {"threshold": 0.0},
    "RefCM": {"discovery_threshold": 0.0},
}

n_list = [5_000, 20_000, 50_000, 100_000, 200_000]
n_types_list = [10, 20, 35, 60, 100]
n_runs_list = [5] * 5

timings = []

for n, n_types, n_runs in zip(n_list, n_types_list, n_runs_list):
    for i in range(n_runs):
        q, ref = generate_upsampled(n_cells=n, n_celltypes=n_types)

        for model in models:
            m = model()
            prep_adata(ref, target_sum=10_000 if m.id_ == "CellTypist" else None)

            if m.id_ in ["SCANVI", "SCALEX"]:
                shutil.rmtree("SCALEX/q", ignore_errors=True)
                shutil.rmtree("SCALEX/q_r", ignore_errors=True)
                shutil.rmtree("SCALEX/r", ignore_errors=True)

            with suppress_all_console():
                m.setref(ref, key, **kwargs)

            prep_adata(q, target_sum=10_000 if m.id_ == "CellTypist" else None)

            with suppress_all_console():
                a = m.annotate(q, key, **kwargs)

            print(f"{m.id_:<15} : {n:>20} | {n_types:<20} | {m.time}")

            timings.append([m.id_, n, n_types, i, m._t_total, m._t_ref, m._t_annot])

            df = pd.DataFrame(
                data=timings,
                columns=[
                    "method",
                    "n",
                    "n_types",
                    "run",
                    "t_total",
                    "t_ref",
                    "t_annot",
                ],
            )
            df.to_csv("timings.csv")
