import os
import json
import numpy as np

from anndata import AnnData
from typing import Dict, TypeAlias


class Annot:
    def __init__(
        self,
        preds: list | None = None,
        ref: AnnData | None = None,
        rkey: str | None = None,
    ) -> None:

        # annotations
        self._preds = None
        self._rlabels = None

        # performance metrics
        self._true = None
        self._qlabels = None
        self._novlabels = None
        self._acc = None
        self._cacc = None  # closed-set acc
        self._cfmatrix = None

        # init if arguments passed
        if preds is not None and ref is not None and rkey is not None:
            self._set_preds(preds, ref, rkey)

    def _set_preds(self, preds: list, ref: AnnData, rkey: str) -> None:
        self._rlabels = sorted(ref.obs[rkey].unique().tolist()) + ["novel"]

        # convert to ints for efficiency
        preds = np.array(preds)
        self._preds = np.zeros(preds.shape, dtype=int)
        for i, rl in enumerate(self._rlabels):
            self._preds[preds == rl] = i

    def _set_true(self, q: AnnData, qkey: str) -> None:
        self._qlabels = sorted(q.obs[qkey].unique().tolist())
        self._novlabels = list(set(self._qlabels) - set(self._rlabels))

        # convert to ints for efficiency
        true = np.array(q.obs[qkey])
        self._true = np.zeros(true.shape, dtype=int)
        for i, t in enumerate(self._qlabels):
            self._true[true == t] = i

    def eval_(self, q: AnnData, qkey: str) -> None:
        self._set_true(q, qkey)
        self._eval()

    def _eval(self) -> None:

        correct = 0
        self._cfmatrix = np.zeros((len(self._qlabels), len(self._rlabels)), dtype=int)

        ccorrect = 0
        closed_mask = np.ones(len(self._true), dtype=bool)
        for nl in self._novlabels:
            closed_mask &= np.array(self.true) != nl
        n_closed = int(closed_mask.sum())

        for i, ql in enumerate(self._qlabels):
            true_mask = self._true == i

            for j, rl in enumerate(self._rlabels):
                pred_mask = self._preds == j

                if (ql == rl) or (ql in self._novlabels and rl == "novel"):
                    correct += (true_mask & pred_mask).sum()

                if (ql == rl) and (ql not in self._novlabels):
                    ccorrect += (true_mask & pred_mask).sum()

                self._cfmatrix[i, j] = (true_mask & pred_mask).sum()

        self._acc = float(correct / len(self._true))
        self._cacc = float(ccorrect / n_closed) if n_closed > 0 else float("nan")

    def copy(self) -> "Annot":
        a = Annot()
        a._preds = None if self._preds is None else self._preds.copy()
        a._rlabels = None if self._rlabels is None else self._rlabels.copy()

        # performance metrics
        a._true = None if self._true is None else self._true.copy()
        a._qlabels = None if self._qlabels is None else self._qlabels.copy()
        a._novlabels = None if self._novlabels is None else self._novlabels.copy()
        a._acc = self._acc
        a._cfmatrix = None if self._cfmatrix is None else self._cfmatrix.copy()

        return a

    @property
    def acc(self) -> float:
        return self._acc

    @property
    def cacc(self) -> float:
        return self._cacc

    @property
    def cfmatrix(self) -> np.ndarray:
        return self._cfmatrix

    @property
    def preds(self) -> list:
        tmp = self._preds.copy().astype(str)

        for i, rl in enumerate(self._rlabels):
            tmp[self._preds == i] = rl

        return tmp.tolist()

    @property
    def true(self) -> list:
        tmp = self._true.copy().astype(str)

        for i, rl in enumerate(self._qlabels):
            tmp[self._true == i] = rl

        return tmp.tolist()

    @property
    def json(self) -> Dict:

        d = {
            "preds": self._preds.tolist(),
            "rlabels": self._rlabels,
        }

        if self._true is None:
            return d
        else:
            return d | {
                "true": self._true.tolist(),
                "qlabels": self._qlabels,
                "novlabels": self._novlabels,
                "acc": self._acc,
                "cacc": self._cacc,
                "cfmatrix": self._cfmatrix.tolist(),
            }

    def __repr__(self) -> str:
        if self._true is not None:
            if self._acc != self._cacc:
                return f"Annot [{self._acc:.2f} | {self._cacc:.2f}]"
            else:
                return f"Annot [{self._acc:.2f}]"
        return "Annot [   ]"


def annot_from_json(d: Dict) -> Annot:
    a = Annot()
    a._preds = np.array(d["preds"])
    a._rlabels = d["rlabels"]

    if "true" in d:
        a._true = np.array(d["true"])
        a._qlabels = d["qlabels"]
        a._novlabels = d["novlabels"]
        a._acc = d["acc"]
        a._cacc = d["cacc"]
        a._cfmatrix = np.array(d["cfmatrix"])
        # a._eval()

    return a


BenchmarkDB: TypeAlias = Dict[str, Dict[str, Dict[str, Annot]]]


def load_benchdb(fpath: str) -> BenchmarkDB:
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            db = json.load(f)
    else:
        db = {}

    for m in db:
        for qid in db[m]:
            for rid in db[m][qid]:
                db[m][qid][rid] = annot_from_json(db[m][qid][rid])

    return db


def add_bench(db: BenchmarkDB, m: str, qid: str, rid: str, a: Annot) -> BenchmarkDB:
    if db.get(m) is None:
        db[m] = {}
    if db[m].get(qid) is None:
        db[m][qid] = {}
    db[m][qid][rid] = a

    return db


def save_benchdb(db: BenchmarkDB, fpath: str) -> BenchmarkDB:
    d = {}
    for m in db:
        d[m] = {}
        for qid in db[m]:
            d[m][qid] = {}
            for rid in db[m][qid]:
                d[m][qid][rid] = db[m][qid][rid].json

    with open(fpath, "w") as f:
        json.dump(d, f)

    return db


def mvote(a: Annot, clustering: np.ndarray | None = None) -> Annot:
    assert clustering is not None or a._true is not None

    clustering = a._true if clustering is None else clustering
    clusters = (
        np.arange(len(a._qlabels)) if clustering is None else np.unique(clustering)
    )

    mv_preds = np.zeros_like(a._preds)

    for c in clusters:
        m = clustering == c

        vals, counts = np.unique(a._preds[m], return_counts=True)
        mv = vals[np.argmax(counts)]

        mv_preds[m] = mv

    # create new majority-vote annotation
    mv = Annot()
    mv._preds = mv_preds
    mv._rlabels = a._rlabels

    if a._true is not None:
        mv._true = a._true
        mv._qlabels = a._qlabels
        mv._novlabels = a._novlabels
        mv._eval()

    return mv


def mvote_all(db: BenchmarkDB, to_mvote: list[str]) -> BenchmarkDB:
    keys = [k for k in db.keys() if any([k == m for m in to_mvote])]

    for m in keys:

        for qid in db[m]:
            for rid in db[m][qid]:
                a = mvote(db[m][qid][rid])
                add_bench(db, f"MV-{m}", qid, rid, a)
