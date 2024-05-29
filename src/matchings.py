import numpy as np
import pandas as pd
from anndata import AnnData
import plotly.express as px
import plotly.graph_objects as go
from typing import Literal, Union, List, Callable, Tuple


# config and logging setup
import config
import logging

logging = logging.getLogger(__name__)

NOTMAPPED = -1
INCORRECT = 0
CORRECT = 1


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

        # self.q_cts_per_label: np.ndarray = np.unique(q.labels, return_counts=True)[1]

    def eval(self, ground_truth_obs_key: str) -> None:
        """
        Evaluates a given query -> reference matching

        Parameters
        ----------
        ground_truth_obs_key: str
            Ground truth '.obs' key for the query dataset
        """
        gt = self.q.obs[ground_truth_obs_key]
        q_ktl = {}
        for i in range(self.q_n):
            lbl = gt[self.q.obs["refcm_clusters"] == i].mode().iloc[0]
            q_ktl[i] = lbl

        q_labels = np.array([*q_ktl.values()])
        label_intersection = sorted([*(set(self.ref_labels).intersection(q_labels))])

        self.q_ktl = q_ktl
        self.label_intersection = label_intersection
        self.n_common_labels = len(label_intersection)

        self.ms: np.ndarray = np.full_like(self.m, NOTMAPPED)  # ms: map status
        for i in range(self.q_n):
            ql = q_ktl[i]

            for j in range(self.r_n):
                rl = self.ref_ktl[j]

                if self.m[i, j] == 1:
                    self.ms[i, j] = CORRECT if ql == rl else INCORRECT
                    s = "\x1b[32m[+]\x1B[0m" if ql == rl else "\x1B[31m[-]\x1B[0m"
                    logging.debug(f"{s} {ql:<20} mapped to {rl:<20}")

        # calculations on the number of correct/incorrect/missing links
        correct_mask = self.ms == CORRECT
        incorrect_mask = self.ms == INCORRECT
        notmapped_mask = self.ms == NOTMAPPED

        self.n_correct_links: int = correct_mask.sum()
        self.n_incorrect_links: int = incorrect_mask.sum()
        self.n_notmapped_links: int = notmapped_mask.sum()

        # useful masks for narrowing down how common/noncommon types are linked
        qc_w_correct_mapping_mask = correct_mask.any(axis=1)
        qc_w_incorrect_mapping_mask = incorrect_mask.any(axis=1)
        qc_wo_mapping_mask = notmapped_mask.all(axis=1)

        cmn_lbl_mask = np.array(
            [*map(lambda x: x in label_intersection, q_ktl.values())]
        )

        # query clusters with at least one correct link
        self.common_w_correct_links = q_labels[
            cmn_lbl_mask & qc_w_correct_mapping_mask
        ].tolist()
        self.n_common_w_correct_links = len(self.common_w_correct_links)

        # query clusters with only correct links
        self.common_w_only_correct_links = q_labels[
            cmn_lbl_mask & qc_w_correct_mapping_mask & ~qc_w_incorrect_mapping_mask
        ].tolist()
        self.n_common_w_only_correct_links = len(self.common_w_only_correct_links)

        # query clusters with incorrect links
        self.common_w_incorrect_links = q_labels[
            cmn_lbl_mask & qc_w_incorrect_mapping_mask
        ].tolist()
        self.n_common_w_incorrect_links = len(self.common_w_incorrect_links)

        # query clusters with only incorrect links
        self.common_w_only_incorrect_links = q_labels[
            cmn_lbl_mask & ~qc_w_correct_mapping_mask & qc_w_incorrect_mapping_mask
        ].tolist()
        self.n_common_w_only_incorrect_links = len(self.common_w_only_incorrect_links)

        # query clusters with a correct link, but also incorrect link(s) from splitting.
        self.common_w_correct_and_incorrect_links = q_labels[
            cmn_lbl_mask & qc_w_correct_mapping_mask & qc_w_incorrect_mapping_mask
        ].tolist()
        self.n_common_w_correct_and_incorrect_links = len(
            self.common_w_correct_and_incorrect_links
        )

        # query clusters that are incorrectly marked as discovered (i.e. not mapped to anything)
        self.common_notmapped = q_labels[cmn_lbl_mask & qc_wo_mapping_mask].tolist()
        self.n_common_notmapped = len(self.common_notmapped)

        # noncommon query clusters that are correctly marked as discovered.
        self.noncommon_discovered = q_labels[
            ~cmn_lbl_mask & qc_wo_mapping_mask
        ].tolist()
        self.n_noncommon_discovered = len(self.noncommon_discovered)

        # noncommon query clusters that are incorrectly assigned to anything
        self.noncommon_mapped = q_labels[(~cmn_lbl_mask & ~qc_wo_mapping_mask)].tolist()
        self.n_noncommon_mapped = len(self.noncommon_mapped)

        # logging
        logging.info(f"{self.q_name:<20} to {self.ref_name:<20}")
        logging.info(f"{self.n_common_labels:<2}    common cell types")
        logging.info(
            f"{self.n_correct_links:<2}/{self.n_common_labels:<2} correct   links"
        )
        logging.info(f"{self.n_incorrect_links:<2}    incorrect links")

    def display_matching_costs(
        self, ground_truth_obs_key: str = None, display_mapped_pairs: bool = True
    ) -> None:
        """
        Displays matching cost matrix between query and reference datasets.

        Parameters
        ----------
        ground_truth_obs_key: str = None
            If available, ground truth '.obs' key for the query dataset
        display_mapped_pairs: bool = False
            Whether to indicate on the cost matrix which pairs ended up paired
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
        fig.update_xaxes(tickangle=-45)

        # add markers indicating which pairs are mapped in the end
        if display_mapped_pairs:
            xs, ys, cs = [], [], []
            for i in range(self.q_n):
                for j in range(self.r_n):
                    if self.m[i, j] == 1:
                        # change the color to indicate whether it is correct or not,
                        # only possible if ground_truth_obs_key was given
                        if ground_truth_obs_key is not None:
                            c = "green" if q_labels[i] == self.ref_ktl[j] else "red"
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
        fig.show()

    def display_matching_graph(self) -> None:
        """Display the bipartite matching graph"""
        # TODO utilize pyviz

        logging.error("Deprecated")
        return

    # def display_sunburst(self, datasets: List[AnnData]) -> None:
    #     """
    #     Displays a sunburst chart comparing distribution & counts of
    #     cell types in each dataset

    #     Parameters
    #     ----------
    #     datasets: List[AnnData]
    #         list of datasets
    #     """
    #     # TODO
    #     dfs = []
    #     for ds in datasets:
    #         df = pd.DataFrame({"key": ds.labels, "source": ds.name})

    #         df["label"] = df.key.apply(lambda s: ds._keys_to_labels[s])
    #         df["key:label"] = "(" + df["key"].astype("str") + ") " + df["label"]
    #         dfs.append(df)
    #     df = pd.concat(dfs)

    #     df2 = (
    #         df[["source", "label"]]
    #         .value_counts()
    #         .reset_index()
    #         .rename(columns={0: "count"})
    #     )
    #     fig = px.sunburst(df2, path=["source", "label"], values="count")
    #     fig.show()

    # def display_count_histogram(datasets: List[AnnData]) -> None:
    #     """
    #     Displays a count histogram comparing distribution & counts of
    #     cell types in each dataset

    #     Parameters
    #     ----------
    #     datasets: List[AnnData]
    #         list of datasets
    #     """
    #     dfs = []
    #     for ds in datasets:
    #         df = pd.DataFrame({"key": ds.labels, "source": ds.name})

    #         df["label"] = df.key.apply(lambda s: ds._keys_to_labels[s])
    #         df["key:label"] = "(" + df["key"].astype("str") + ") " + df["label"]
    #         dfs.append(df)
    #     df = pd.concat(dfs)

    #     fig = px.histogram(
    #         df.sort_values("label"), x="label", color="source", barmode="group"
    #     )
    #     fig.show()
