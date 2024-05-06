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

        q_labels = [*q_ktl.values()]
        label_intersection = [*(set(self.ref_labels).intersection(q_labels))]

        self.map_status: np.ndarray = np.full_like(self.m, NOTMAPPED)
        for i in range(self.q_n):
            ql = q_ktl[i]

            for j in range(self.r_n):
                rl = self.ref_ktl[j]

                if self.m[i, j] == 1:
                    logging.debug(f"{ql:<20} mapped to {rl:<20}")
                    self.map_status[i, j] = CORRECT if ql == rl else INCORRECT

        correct_mask = self.map_status == CORRECT
        incorrect_mask = self.map_status == INCORRECT
        notmapped_mask = self.map_status == NOTMAPPED

        self.n_correct: int = correct_mask.sum()
        self.n_incorrect: int = incorrect_mask.sum()
        self.n_notmapped: int = notmapped_mask.sum()

        # TODO this will need to be modified if we allow multiple "true" clusters
        # self.pct_correct: float = self.q_cts_per_label[correct_mask].sum()
        # self.pct_correct /= self.q_cts_per_label.sum()

        # this is definitely not informative with cluster splitting
        # self.pct_incorrect: float = self.q_cts_per_label[incorrect_mask].sum()
        # self.pct_incorrect /= self.q_cts_per_label.sum()
        # self.pct_notmapped: float = 1 - self.pct_correct - self.pct_incorrect

        logging.info(f"mapped {self.q_name:<20} to {self.ref_name:<20}")
        logging.info(f"({len(label_intersection):<2} common cell types)")
        logging.info(f"{self.n_correct:<2}/{self.q_n:<2} correct mappings")
        logging.info(f"{self.n_incorrect:<2}/{self.q_n:<2} incorrect mappings")

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

        logging.error("Deprecated")
        return
        # TODO just use pyviz instead since this will be much nicer also if we want to display the effect of using multiple references (and saves a bunch of headaches)

        q_x, r_x = 0, max(self.q_n, self.ref_n) / 2
        y_scaling = 1

        q_colors = [
            "#00d2ff" if self.q_ktl[i] in self.ref_labels else "#B3CCF5"
            for i in range(self.q_n)
        ]

        r_colors = [
            "#00d2ff" if self.ref_ktl[i] in self.q_labels else "#B3CCF5"
            for i in range(self.ref_n)
        ]

        q_trace = go.Scatter(
            x=[q_x] * self.q_n,
            y=[(self.q_n / 2 - i) * y_scaling for i in range(self.q_n)],
            text=self.q_labels,
            mode="markers+text",
            hoverinfo="text",
            textposition="top right",
            marker=dict(size=14, color=q_colors),
        )

        r_trace = go.Scatter(
            x=[r_x] * self.ref_n,
            y=[(self.ref_n / 2 - i) * y_scaling for i in range(self.ref_n)],
            text=self.ref_labels,
            mode="markers+text",
            hoverinfo="text",
            textposition="top left",
            marker=dict(size=14, color=r_colors),
        )

        e = {"correct": {"x": [], "y": []}, "incorrect": {"x": [], "y": []}}

        for i in range(self.m.shape[0]):
            for j in range(self.m.shape[1]):
                if self.map_status[i, j] != NOTMAPPED:
                    xs = [q_x, r_x, None]
                    ys = [
                        (self.q_n / 2 - i) * y_scaling,
                        (self.ref_n / 2 - j) * y_scaling,
                        None,
                    ]
                    if self.map_status[i, j] == CORRECT:
                        e["correct"]["x"] += xs
                        e["correct"]["y"] += ys
                    else:
                        e["incorrect"]["x"] += xs
                        e["incorrect"]["y"] += ys

        correct_edge_trace = go.Scatter(
            x=e["correct"]["x"],
            y=e["correct"]["y"],
            line=dict(width=3, color="#90EE90"),
            hoverinfo="none",
            mode="lines",
            name="correct matching",
        )

        incorrect_edge_trace = go.Scatter(
            x=e["incorrect"]["x"],
            y=e["incorrect"]["y"],
            line=dict(width=3, color="#F72F35"),
            hoverinfo="none",
            mode="lines",
            name="incorrect matching",
        )

        fig = go.Figure(
            data=[
                correct_edge_trace,
                incorrect_edge_trace,
                q_trace,
                r_trace,
            ],
            layout=go.Layout(
                title=f"{self.q_name} to {self.ref_name} matching",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        fig.show()

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
