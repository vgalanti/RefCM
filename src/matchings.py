import json
import numpy as np
import pandas as pd
from anndata import AnnData
import plotly.express as px
import plotly.graph_objects as go
from typing import Literal, Union, List, Callable, Tuple, Dict


# config and logging setup
import config
import logging

logging = logging.getLogger(__name__)

NOTMAPPED = -1
INCORRECT = 0
CORRECT = 1

TYPE_EQUALITY_STRICTNESS = 1.0


class Matching:
    def __init__(
        self,
        q: AnnData,
        ref: Union[List[AnnData], AnnData],
        q_name: str = None,
        ref_names: Union[List[str], str] = None,
    ):
        """
        Creates an instance of a Matching object.

        Parameters
        ----------
        q: AnnData
            query dataset
        ref: Union[List[AnnData], AnnData]
            reference dataset(s)
        q_name: str = None
            query dataset name (for graph titles)
        ref_names: Union[List[str], str] = None
            reference dataset(s) name(s) (for graph titles)


        NOTE
        ----
        * ideally, ground truth labels follow the same clustering as the one that was given for matching, meaning that no mapped cluster can have samples with different ground truth values. Otherwise, it will determine "correctness" by what the most frequent value per cluster was in the ground truth column.
        """

        if not isinstance(ref, list):
            ref = [ref]
        if not isinstance(ref_names, list) and ref_names is not None:
            ref_names = [ref_names]

        self.q: AnnData = q
        self.ref: List[AnnData] = ref

        # copied to not get overwritten if refcm is
        # run on the same query multiple times in a row
        self.m: np.ndarray = q.uns["refcm_mapping"].copy()
        self.m_costs: np.ndarray = q.uns["refcm_costs"].copy()

        self.q_n, self.r_n = self.m.shape
        self.q_name: str = q_name
        self.ref_name: str = (
            "-".join(sorted([r for r in ref_names])) if ref_names is not None else None
        )
        self.ref_ktl: dict[int, str] = q.uns["refcm_ref_ktl"].copy()
        self.ref_labels: List[str] = [*self.ref_ktl.values()]

        # create type tree
        self.tree = {}
        self.strictness = TYPE_EQUALITY_STRICTNESS
        with open(config.TREE_FILE, "r") as f:
            tt = json.load(f)
        for p, c, s in tt:
            self.tree[c.lower().strip()] = (p.lower().strip(), s)

    def eval(self, ground_truth_obs_key: str) -> Dict:
        """
        Evaluates a given query -> reference matching

        Parameters
        ----------
        ground_truth_obs_key: str
            Ground truth '.obs' key for the query dataset

        Returns
        -------
        Dict:
            Dictionary containing performance metrics.
        """
        gt = self.q.obs[ground_truth_obs_key]

        # establish correspondence between ground truth label and refcm cluter
        self.q_ktl = {}
        for i in range(self.q_n):
            lbl = gt[self.q.obs["refcm_clusters"] == i].mode().iloc[0]
            self.q_ktl[i] = lbl

        self.q_labels = np.array([*self.q_ktl.values()])
        self.common_labels = sorted(
            [*(set(self.ref_labels).intersection(self.q_labels))]
        )
        self.n_common_labels = len(self.common_labels)

        self.ms: np.ndarray = np.full_like(self.m, NOTMAPPED)  # ms: map status
        for i in range(self.q_n):
            ql = self.q_ktl[i]

            for j in range(self.r_n):
                rl = self.ref_ktl[j]

                if self.m[i, j] == 1:
                    el = self.eval_link(ql, rl)
                    is_correct = el >= self.strictness

                    self.ms[i, j] = CORRECT if is_correct else INCORRECT
                    s = (
                        f"\x1b[32m[+|{el:.2f}]\x1B[0m"
                        if is_correct
                        else f"\x1B[31m[-|{el:.2f}]\x1B[0m"
                    )
                    logging.debug(f"{s} {ql:<20} mapped to {rl:<20}")

        # calculations on the total number of correct/incorrect/missing links
        correct_mask = self.ms == CORRECT
        incorrect_mask = self.ms == INCORRECT
        notmapped_mask = self.ms == NOTMAPPED

        self.n_correct_links: int = correct_mask.sum()
        self.n_incorrect_links: int = incorrect_mask.sum()
        self.n_notmapped_links: int = notmapped_mask.sum()

        # masks used in determining how common/noncommon types are linked
        cmn_lbl_mask = np.array(
            [*map(lambda x: x in self.common_labels, self.q_ktl.values())]
        )

        qc_w_correct_mapping_mask = correct_mask.any(axis=1)
        qc_w_incorrect_mapping_mask = incorrect_mask.any(axis=1)
        qc_wo_mapping_mask = notmapped_mask.all(axis=1)

        cmn_w_correct_links_mask = cmn_lbl_mask & qc_w_correct_mapping_mask
        cmn_w_only_correct_links_mask = (
            cmn_lbl_mask & qc_w_correct_mapping_mask & ~qc_w_incorrect_mapping_mask
        )
        cmn_w_incorrect_links_mask = cmn_lbl_mask & qc_w_incorrect_mapping_mask
        cmn_w_only_incorrect_links_mask = (
            cmn_lbl_mask & ~qc_w_correct_mapping_mask & qc_w_incorrect_mapping_mask
        )
        cmn_w_correct_and_incorrect_links_mask = (
            cmn_lbl_mask & qc_w_correct_mapping_mask & qc_w_incorrect_mapping_mask
        )
        cmn_notmapped_mask = cmn_lbl_mask & qc_wo_mapping_mask
        ncmn_discovered_mask = ~cmn_lbl_mask & qc_wo_mapping_mask
        ncmn_mapped_mask = ~cmn_lbl_mask & ~qc_wo_mapping_mask

        # count of query cells that are/are not common to both query and reference
        qc_cts = np.unique(gt, return_counts=True)[1]
        q_ct_cmn = qc_cts[cmn_lbl_mask].sum()
        q_ct_ncmn = qc_cts[~cmn_lbl_mask].sum()

        # query clusters with at least one correct link
        self.common_w_correct_links = self.q_labels[cmn_w_correct_links_mask].tolist()
        self.n_common_w_correct_links = len(self.common_w_correct_links)
        self.pct_common_w_correct_links = (
            (qc_cts[cmn_w_correct_links_mask].sum() / q_ct_cmn)
            if q_ct_cmn != 0
            else 1.0
        )

        # query clusters with only correct links
        self.common_w_only_correct_links = self.q_labels[
            cmn_w_only_correct_links_mask
        ].tolist()
        self.n_common_w_only_correct_links = len(self.common_w_only_correct_links)
        self.pct_common_w_only_correct_links = (
            (qc_cts[cmn_w_only_correct_links_mask].sum() / q_ct_cmn)
            if q_ct_cmn != 0
            else 1.0
        )

        # query clusters with incorrect links
        self.common_w_incorrect_links = self.q_labels[
            cmn_w_incorrect_links_mask
        ].tolist()
        self.n_common_w_incorrect_links = len(self.common_w_incorrect_links)
        self.pct_common_w_incorrect_links = (
            (qc_cts[cmn_w_incorrect_links_mask].sum() / q_ct_cmn)
            if q_ct_cmn != 0
            else 1.0
        )

        # query clusters with only incorrect links
        self.common_w_only_incorrect_links = self.q_labels[
            cmn_w_only_incorrect_links_mask
        ].tolist()
        self.n_common_w_only_incorrect_links = len(self.common_w_only_incorrect_links)
        self.pct_common_w_only_incorrect_links = (
            (qc_cts[cmn_w_only_incorrect_links_mask].sum() / q_ct_cmn)
            if q_ct_cmn != 0
            else 1.0
        )

        # query clusters with a correct link, but also incorrect link(s) from splitting.
        self.common_w_correct_and_incorrect_links = self.q_labels[
            cmn_w_correct_and_incorrect_links_mask
        ].tolist()
        self.n_common_w_correct_and_incorrect_links = len(
            self.common_w_correct_and_incorrect_links
        )
        self.pct_common_w_correct_and_incorrect_links = (
            (qc_cts[cmn_w_correct_and_incorrect_links_mask].sum() / q_ct_cmn)
            if q_ct_cmn != 0
            else 1.0
        )

        # query clusters that are incorrectly marked as discovered (i.e. not mapped to anything)
        self.common_notmapped = self.q_labels[cmn_notmapped_mask].tolist()
        self.n_common_notmapped = len(self.common_notmapped)
        self.pct_common_notmapped = (
            qc_cts[cmn_notmapped_mask].sum() / q_ct_cmn if q_ct_cmn != 0 else 1.0
        )

        # noncommon query clusters that are correctly marked as discovered.
        self.noncommon_discovered = self.q_labels[ncmn_discovered_mask].tolist()
        self.n_noncommon_discovered = len(self.noncommon_discovered)
        self.pct_noncommon_discovered = (
            qc_cts[ncmn_discovered_mask].sum() / q_ct_ncmn if q_ct_ncmn != 0 else 1.0
        )

        # noncommon query clusters that are incorrectly assigned to anything
        self.noncommon_mapped = self.q_labels[ncmn_mapped_mask].tolist()
        self.n_noncommon_mapped = len(self.noncommon_mapped)
        self.pct_noncommon_mapped = (
            qc_cts[ncmn_mapped_mask].sum() / q_ct_ncmn if q_ct_ncmn != 0 else 1.0
        )

        # logging
        logging.info(f"{self.q_name:<20} to {self.ref_name:<20}")
        logging.info(f"{self.n_common_labels:<2}    common cell types")
        logging.info(
            f"{self.n_correct_links:<2}/{self.n_common_labels:<2} correct   links"
        )
        logging.info(f"{self.n_incorrect_links:<2}    incorrect links")

        # return summary stats as dictionary
        return {
            "common": self.common_labels,
            "n_common": len(self.common_labels),
            "counts": {k: v for k, v in zip(*np.unique(gt, return_counts=True))},
            "counts_total": qc_cts.sum(),
            "counts_total_common": q_ct_cmn,
            "counts_total_noncommon": q_ct_ncmn,
            "n_correct_links": self.n_correct_links,
            "n_incorrect_links": self.n_incorrect_links,
            "n_notmapped_links": self.n_notmapped_links,
            "common_w_correct_links": self.common_w_correct_links,
            "n_common_w_correct_links": self.n_common_w_correct_links,
            "pct_common_w_correct_links": self.pct_common_w_correct_links,
            "common_w_only_correct_links": self.common_w_only_correct_links,
            "n_common_w_only_correct_links": self.n_common_w_only_correct_links,
            "pct_common_w_only_correct_links": self.pct_common_w_only_correct_links,
            "common_w_incorrect_links": self.common_w_incorrect_links,
            "n_common_w_incorrect_links": self.n_common_w_incorrect_links,
            "pct_common_w_incorrect_links": self.pct_common_w_incorrect_links,
            "common_w_only_incorrect_links": self.common_w_only_incorrect_links,
            "n_common_w_only_incorrect_links": self.n_common_w_only_incorrect_links,
            "pct_common_w_only_incorrect_links": self.pct_common_w_only_incorrect_links,
            "common_w_correct_and_incorrect_links": self.common_w_correct_and_incorrect_links,
            "n_common_w_correct_and_incorrect_links": self.n_common_w_correct_and_incorrect_links,
            "pct_common_w_correct_and_incorrect_links": self.pct_common_w_correct_and_incorrect_links,
            "common_notmapped": self.common_notmapped,
            "n_common_notmapped": self.n_common_notmapped,
            "pct_common_notmapped": self.pct_common_notmapped,
            "noncommon_discovered": self.noncommon_discovered,
            "n_noncommon_discovered": self.n_noncommon_discovered,
            "pct_noncommon_discovered": self.pct_noncommon_discovered,
            "noncommon_mapped": self.noncommon_mapped,
            "n_noncommon_mapped": self.n_noncommon_mapped,
            "pct_noncommon_mapped": self.pct_noncommon_mapped,
        }

    def display_matching_costs(
        self,
        ground_truth_obs_key: str = None,
        display_mapped_pairs: bool = True,
        show_all_labels: bool = False,
        show_values: bool = False,
        angle_x_labels: bool = False,
        width: float = -1,
        height: float = -1,
    ) -> None:
        """
        Displays matching cost matrix between query and reference datasets.

        Parameters
        ----------
        ground_truth_obs_key: str = None
            If available, ground truth '.obs' key for the query dataset
        display_mapped_pairs: bool = False
            Whether to indicate on the cost matrix which pairs ended up paired
        show_all_labels: bool = False
            Whether to show all labels (types) in the x and y axes
        show_values: bool = False
            Whether to show the values (without hovering)
        angle_x_labels: bool = False
            Whether to angle the x labels at 45 deg instead of 90 deg
        width: float = -1
            If positive, figure width
        height: float = -1
            If positive, figure height
        """
        title = (
            f"{self.q_name} to {self.ref_name} matching cost"
            if self.q_name is not None and self.ref_name is not None
            else ""
        )

        y_label = "Query" if self.q_name is None else f"Query: {self.q_name}"
        x_label = (
            "Reference" if self.ref_name is None else f"Reference: {self.ref_name}"
        )

        if ground_truth_obs_key is not None:
            gt = self.q.obs[ground_truth_obs_key]
            q_labels = []
            for i in range(self.q_n):
                lbl = gt[self.q.obs["refcm_clusters"] == i].mode().iloc[0]
                q_labels.append(lbl)
        else:
            q_labels = list(range(self.q_n))

        # plot the matching costs
        fig = px.imshow(
            self.m_costs,
            title=title,
            labels=dict(y=y_label, x=x_label, color="cost"),
            x=self.ref_labels,
            y=q_labels,
            color_continuous_scale="Agsunset",
        )
        fig.update_xaxes(tickangle=-90)

        if show_values:
            for i in range(self.q_n):
                for j in range(self.r_n):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{self.m_costs[i][j]:.2f}",
                        showarrow=False,
                        # bgcolor="white",
                        # opacity=0.2,
                    )
            fig.update_annotations(font=dict(color="white"))  # size=30

        # add markers indicating which pairs are mapped in the end
        if display_mapped_pairs:
            xs, ys, cs = [], [], []
            for i in range(self.q_n):
                for j in range(self.r_n):
                    if self.m[i, j] == 1:
                        # change the color to indicate whether it is correct or not,
                        # only possible if ground_truth_obs_key was given
                        if ground_truth_obs_key is not None:
                            c = (
                                "green"
                                if self.eval_link(q_labels[i], self.ref_ktl[j])
                                >= self.strictness
                                else "red"
                            )

                        else:
                            c = "blue"

                        xs.append(self.ref_ktl[j])
                        ys.append(q_labels[i])
                        cs.append(c)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(color=cs, size=5),
                    name="",
                    hovertemplate="%{y} -> %{x}",
                )
            )

        # final tweaks
        if angle_x_labels:
            fig.update_xaxes(tickangle=-45)
        if width > 0:
            fig.update_layout(width=width)
        if height > 0:
            fig.update_layout(height=height)
        if show_all_labels:
            fig.update_yaxes(dtick=1)
            fig.update_xaxes(dtick=1)

        fig.show()

    def eval_link(self, q_label: str, ref_label: str) -> float:
        """
        Evaluates the matching/link between query and reference labels/types.


        Parameters
        ----------
        q_label: str
            The query label to check for equality.
        ref_label: str
            The reference label to check with.

        Returns
        -------
        float:
            Number between 0 and 1 that indicates how equal the labels are.
            1 if and only if this is an exact match, e.g. if the labels match
            up to upper/lower-case and whitespaces differences, or it's just an
            equivalent choice from the authors (i.e. "Treg" vs "Regulatory T cell").
            0 indicates that the labels are completely different.
        """
        # make inputs lowercase, and remove whitespaces
        q_label = q_label.lower().strip()
        ref_label = ref_label.lower().strip()

        # check if the two are directly equal
        if q_label == ref_label:
            return 1.0

        # otherwise, go through ancestors to check closest parent
        q_anc, q_sim = self._tree_ancestors(q_label)
        ref_anc, ref_sim = self._tree_ancestors(ref_label)

        isect = [i for i, v in enumerate(q_anc) if v in ref_anc]

        if len(isect) == 0:
            return 0

        q_idx = min(isect)
        ref_idx = ref_anc.index(q_anc[q_idx])

        return min(q_sim[q_idx], ref_sim[ref_idx])

    def _tree_ancestors(self, t: str) -> Tuple[List[str], List[float]]:
        ancestors = [t]
        sim = [1.0]
        while self.tree.get(t) is not None:
            t, v = self.tree.get(t)
            ancestors.append(t)
            sim.append(v)

        return ancestors, sim

    def set_type_equality_strictness(self, t: float) -> None:
        if t < 0 or t > 1:
            logging.error(f"Type strictness must be between 0 and 1.")
        self.strictness = t
