import gc
import shutil
import numpy as np
import benchdb as bdb
import benchutils as bu

from benchmodels import *
from sklearn.model_selection import GroupKFold


db = bdb.load_benchdb("tmuris_drop.json")
models = [
    RefCM,
    Clustifyr,
    SingleRcluster,
    CIPR,
    CellTypist,
    SVM,
    Seurat,
    SingleR,
    SCMAPCell,
    SCMAPCluster,
    SCANVI,
    SCALEX,
]

kwargs = {
    "scmapcell": {"w_agree": 1, "threshold": 0.0},
    "scmapcluster": {"threshold": 0.0},
    "RefCM": {"discovery_threshold": 0.0},
}


ds = bu.load_adata("../data/tmuris_senis_droplet.h5ad")
key = "cell_ontology_class"
split_on = "mouse.id"


groups = ds.obs["mouse.id"].copy()
X = np.arange(ds.shape[0])
y = ds.obs[key].copy()

del ds
gc.collect()

gkf = GroupKFold(n_splits=5)
for fold, (ref_idx, q_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    q_groups = " ".join(list(np.unique(groups[q_idx])))
    ref_groups = " ".join(list(np.unique(groups[ref_idx])))

    print(f"fold {fold}")
    print(f"  q   groups ({len(q_idx):<5}) : {q_groups}")
    print(f"  ref groups ({len(ref_idx):<5}) : {ref_groups}")

    # re-load and split data
    ref = bu.load_adata("../data/tmuris_senis_droplet.h5ad")

    q = ref[q_idx].copy()
    ref = ref[ref_idx].copy()

    for model in models:

        m = model()

        q = bu.prep_adata(q, target_sum=10_000 if m.id_ == "CellTypist" else None)
        ref = bu.prep_adata(ref, target_sum=10_000 if m.id_ == "CellTypist" else None)

        if m.id_ in ["scANVI", "SCALEX"]:
            shutil.rmtree("SCALEX/q", ignore_errors=True)
            shutil.rmtree("SCALEX/q_r", ignore_errors=True)
            shutil.rmtree("SCALEX/r", ignore_errors=True)

        with suppress_all_console():
            m.setref(ref, key, **kwargs)
            a = m.annotate(q, key, **kwargs)

        # R side cleanup
        ro.globalenv.clear()
        ro.r("gc()")

        # eval
        a.eval_(q, key)
        mv = bdb.mvote(a)

        print(f"{m.id_:<15} | cacc {a.cacc:.3f} | mv-cacc: {mv.cacc:.3f} | {m.time}")

        bdb.add_bench(db, m.id_, f"fold_{fold}", "fref", a)
        bdb.save_benchdb(db, "tmuris_kfold.json")
