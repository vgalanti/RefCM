import copy
import numpy as np
import pandas as pd
import scanpy as sc
import plotly.express as px
import plotly.colors as pc
import plotly.graph_objects as go

from scanpy import AnnData
from benchdb import BenchmarkDB, Annot
from benchmodels import RM_METHODS, RCM_METHODS, MV_METHODS
from plotly.subplots import make_subplots


""" --------------- utils ---------------"""


def bench_to_df(bench: BenchmarkDB) -> pd.DataFrame:
    data = []

    for m in bench:
        for q in bench[m]:
            for r in bench[m][q]:

                mtype = (
                    "MV" if m in MV_METHODS else ("RM" if m in RM_METHODS else "RCM")
                )

                if q.startswith("pancreas"):
                    dataset = "scIB pancreas"
                elif q.startswith("pbmc"):
                    dataset = "PBMC Bench1"
                elif q.startswith("mag"):
                    dataset = "Adrenal"
                elif q.startswith("cellbench"):
                    dataset = "CellBench"
                elif q in ["ALM", "MTG", "VISp"]:
                    dataset = "Allen Brain"
                elif q in ["frog", "zebrafish"]:
                    dataset = "Embryo"
                elif q.startswith("fold_"):
                    dataset = "Tabula Muris Senis"
                else:
                    dataset = "other"

                def beautify(s):
                    s = s.replace("pancreas_", "").replace("pbmc_", "")
                    s = (
                        s.replace("mag_", "")
                        .replace("cellbench_", "")
                        .replace("_5cl", "")
                        .replace("fold_", "fold ")
                    )
                    return s

                bq = beautify(q)
                br = beautify(r)

                task = f"{bq} | {br}" if r != "fref" else bq

                data.append([m, mtype, dataset, bq, br, task, bench[m][q][r].cacc])

    df = pd.DataFrame(
        data, columns=["method", "mtype", "dataset", "q", "ref", "task", "acc"]
    )
    df["mtype"] = pd.Categorical(
        df["mtype"], categories=["RM", "MV", "RCM"], ordered=True
    )

    return df


""" --------------- fig styling ---------------"""


def add_paper_styling(fig, lines: bool = True, showgrid: bool = False):

    fig.update_layout(
        template="simple_white",
        legend=dict(
            title="",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,  # push just outside the axes
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            itemsizing="constant",
        ),
    )

    if lines:
        fig.update_traces(line=dict(width=2))  # consistent stroke

    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        showgrid=showgrid,
        zeroline=False,
        mirror=True,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        showgrid=showgrid,
        zeroline=False,
        mirror=True,
    )

    return fig


""" --------------- UMAP ---------------"""


def umap(adata: AnnData, ckey: str, msize: int = 3, **kwargs):

    # info to plot
    x = adata.obsm["X_umap"][:, 0]
    y = adata.obsm["X_umap"][:, 1]
    df = pd.DataFrame({"x": x, "y": y, "color": adata.obs[ckey]}).sort_values("color")

    # plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="color",
        color_discrete_sequence=sc.plotting.palettes.default_102,
    )

    fig.update_layout(
        plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10), **kwargs
    )
    fig.update_traces(marker=dict(size=msize))
    fig.update_legends(
        itemsizing="constant",
        title=None,
        orientation="v",
        xanchor="left",
        yanchor="top",
    )

    px_pad = 0.01  # 1% padding
    xpad = (x.max() - x.min()) * px_pad
    ypad = (y.max() - y.min()) * px_pad

    add_paper_styling(fig, showgrid=False)

    fig.update_xaxes(
        showticklabels=False,
        title=None,
        tickwidth=0,
        ticklen=0,
        range=[x.min() - xpad, x.max() + xpad],
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        tickwidth=0,
        ticklen=0,
        showticklabels=False,
        title=None,
        range=[y.min() - ypad, y.max() + ypad],
    )

    return fig


def umap_acc(adata: AnnData, annot: Annot, model: str = None, msize: int = 3, **kwargs):
    # info to display
    adata.obs["_preds"] = annot.preds
    adata.obs["_true"] = annot.true

    x = adata.obsm["X_umap"][:, 0]
    y = adata.obsm["X_umap"][:, 1]
    df = pd.DataFrame({"x": x, "y": y})

    colors = (
        (adata.obs["_preds"] == adata.obs["_true"])
        .astype(int)
        .map({0: "crimson", 1: "lightgreen"})
    )

    # plot
    title = (
        "" if model is None else f"{model} <br><sup>accuracy: {annot.cacc:.2f} </sup>"
    )
    fig = px.scatter(df, x="x", y="y", title=title)
    fig.update_layout(
        plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10), **kwargs
    )
    fig.update_traces(marker=dict(size=msize, color=colors))
    fig.update_legends(
        itemsizing="constant",
        title=None,
        orientation="v",
        xanchor="left",
        yanchor="top",
    )

    px_pad = 0.01  # 1% padding
    xpad = (x.max() - x.min()) * px_pad
    ypad = (y.max() - y.min()) * px_pad

    add_paper_styling(fig, showgrid=False)

    fig.update_xaxes(
        showticklabels=False,
        title=None,
        tickwidth=0,
        ticklen=0,
        range=[x.min() - xpad, x.max() + xpad],
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        tickwidth=0,
        ticklen=0,
        showticklabels=False,
        title=None,
        range=[y.min() - ypad, y.max() + ypad],
    )

    adata.obs.drop(columns=["_preds", "_true"], inplace=True)

    return fig


""" --------------- pairwise annotation ---------------"""


def plot_pairwise(
    df: pd.DataFrame,
    method: str,
    dataset: str,
    zmin: float = 1.0,
    title: str = "",
    ytitle: str = "Query",
    xtitle: str = "Reference",
    text_auto: bool = True,
    color_continuous_scale="Blues",
) -> None:

    # Filter to method and dataset
    dff = df[(df["method"] == method) & (df["dataset"] == dataset)]

    if dff.empty:
        methods = df["method"].unique().tolist()
        datasets = df["dataset"].unique().tolist()
        print(f"Couldn't find entries for {method} and {dataset}")
        print(f"valid methods are {methods}")
        print(f"valid datasets are {datasets}")
        return None

    # Pivot to matrix form: rows=q, cols=ref, values=acc
    mat = dff.pivot_table(index="q", columns="ref", values="acc", aggfunc="first")

    # Ensure square matrix with consistent ordering
    elements = mat.index.union(mat.columns).tolist()
    mat = mat.reindex(index=elements, columns=elements)

    data = mat.round(2).values
    _zmin = np.nanmin(data)

    if zmin > _zmin:
        zmin = _zmin

    fig = px.imshow(
        data,
        title=title,
        color_continuous_scale=color_continuous_scale,
        labels=dict(x=xtitle, y=ytitle),
        x=elements,
        zmin=zmin,
        zmax=1,
        y=elements,
        width=1000,
        height=400,
        text_auto=text_auto,
    )

    return fig


def plot_pairwise_panel(
    df: pd.DataFrame,
    methods: list[str],
    dataset: str,
    rows=1,
    cols=None,
    zmin=0,
    zmax=1,
    title="Pairwise annotation accuracies",
    ytitle="Query",
    xtitle="Reference",
    fs=16,
    shared_colorbar=True,
    colorbar_title="Accuracy",
    show_outer_ticks=True,
    show_all_ticks=False,
    remove_tick_marks=True,
    title_x_off=0.37,
    title_y_off=0.98,
    ytitle_x_off=-0.13,
    ytitle_y_off=0.5,
    xtitle_x_off=0.5,
    xtitle_y_off=-0.18,
    cell_w=280,
    cell_h=260,
    hspace=0.03,
    vspace=0.08,
):

    figs = []
    for m in methods:
        f = plot_pairwise(df, m, dataset, zmin=zmin)
        figs.append(f)

    n = len(figs)
    if cols is None:
        cols = n
    assert rows * cols >= n, "insufficient rows/cols"

    blues = pc.sequential.Blues

    # Grab canonical labels from first fig's trace (px.imshow sets these)
    tr0 = figs[0].data[0]
    x_labels = list(tr0.x) if getattr(tr0, "x", None) is not None else None
    y_labels = list(tr0.y) if getattr(tr0, "y", None) is not None else None

    panel = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=methods,
        horizontal_spacing=hspace,
        vertical_spacing=vspace,
    )

    for k, f in enumerate(figs):
        r = (k // cols) + 1
        c = (k % cols) + 1

        tr = copy.deepcopy(f.data[0])

        # Detach from px.imshow's layout.coloraxis
        if hasattr(tr, "coloraxis"):
            tr.coloraxis = None

        tr.update(colorscale=blues, zmin=zmin, zmax=zmax, zauto=False, showscale=False)

        # Shared colorbar: put on last trace only
        if shared_colorbar and (k == n - 1):
            tr.showscale = True
            tr.colorbar = dict(
                title=colorbar_title, x=1.02, y=0.5, len=0.9, thickness=12
            )

        panel.add_trace(tr, row=r, col=c)

    panel.update_layout(coloraxis=None)

    panel.update_layout(
        width=cell_w * cols + (90 if shared_colorbar else 30),
        height=cell_h * rows + 60,
        margin=dict(l=90, r=(90 if shared_colorbar else 20), t=40, b=90),
    )

    panel.update_yaxes(autorange="reversed")

    add_paper_styling(panel, lines=False)

    for k, f in enumerate(figs):
        r = (k // cols) + 1
        c = (k % cols) + 1

        # Decide where tick labels should appear
        if show_all_ticks:
            show_x = True
            show_y = True
        elif show_outer_ticks:
            show_x = r == rows  # bottom row only
            show_y = c == 1  # left column only
        else:
            show_x = False
            show_y = False

        # Apply axis formatting
        panel.update_xaxes(
            showticklabels=show_x,
            tickmode="array" if x_labels is not None else None,
            tickvals=x_labels if x_labels is not None else None,
            ticktext=x_labels if x_labels is not None else None,
            tickangle=45,
            showgrid=False,
            zeroline=False,
            ticks="" if remove_tick_marks else "outside",
            ticklen=0 if remove_tick_marks else 5,
            row=r,
            col=c,
        )

        panel.update_yaxes(
            showticklabels=show_y,
            tickmode="array" if y_labels is not None else None,
            tickvals=y_labels if y_labels is not None else None,
            ticktext=y_labels if y_labels is not None else None,
            showgrid=False,
            zeroline=False,
            ticks="" if remove_tick_marks else "outside",
            ticklen=0 if remove_tick_marks else 5,
            row=r,
            col=c,
        )

    # x title (bottom center)
    panel.add_annotation(
        text=xtitle,
        x=xtitle_x_off,
        y=xtitle_y_off,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=fs),
    )
    # y title (left center, rotated)
    panel.add_annotation(
        text=ytitle,
        x=ytitle_x_off,
        y=ytitle_y_off,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=dict(size=fs),
    )

    # main title
    panel.update_layout(
        title=dict(
            text=title,
            x=title_x_off,
            xanchor="right",
            y=title_y_off,
            yanchor="top",
            font=dict(size=fs + 4),
        )
    )

    for ann in panel.layout.annotations:
        if ann.text in methods:
            ann.yshift = (ann.yshift or 0) + 3

    panel.update_layout(margin=dict(l=140, b=140, r=90, t=90))

    return panel


""" --------------- Box plots ---------------"""


def plot_box(
    df: pd.DataFrame,
    dataset: str,
    methods: list[str] = None,
    reorder_by_median: bool = True,
    merge_mv_rcm: bool = True,
    color_boxes: bool = True,
    horizontal: bool = True,
    gray_box_fill: str = "rgba(180, 180, 180, 0.28)",
    gray_point: str = "rgba(90, 90, 90, 0.55)",
):
    # Filter to dataset
    dff = df[df["dataset"] == dataset].copy()

    if dff.empty:
        datasets = df["dataset"].unique().tolist()
        print(f"Couldn't find entries for {dataset}")
        print(f"valid datasets are {datasets}")
        return None

    # Filter to requested methods
    if methods is not None:
        dff = dff[dff["method"].isin(methods)]

    if merge_mv_rcm:
        dff.loc[dff["mtype"] == "MV", "mtype"] = "RCM"

    # Determine method ordering
    if reorder_by_median:
        method_order = (
            dff.groupby("method")["acc"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        method_order = [m for m in methods if m in dff["method"].unique()]

    dff["method"] = pd.Categorical(dff["method"], categories=method_order, ordered=True)
    dff = dff.sort_values("method")

    x, y = ("acc", "method") if horizontal else ("method", "acc")

    fig = px.box(
        dff,
        x=x,
        y=y,
        color="mtype" if color_boxes else None,
        points="all",
        category_orders={
            "method": method_order,
            "mtype": ["RCM", "MV", "RM"],
        },
    )

    add_paper_styling(fig)

    STYLE = {
        "RCM": dict(marker="rgba(0, 60, 140, 0.95)", fill="rgba(0, 60, 140, 0.25)"),
        "MV": dict(marker="rgba(170, 90, 0, 0.92)", fill="rgba(170, 90, 0, 0.25)"),
        "RM": dict(marker="rgba(0, 0, 0, 0.50)", fill="rgba(0, 0, 0, 0.25)"),
    }

    for tr in fig.data:
        if color_boxes and (st := STYLE.get(tr.name)):
            tr.marker.color = st["marker"]
            tr.fillcolor = st["fill"]
        else:
            tr.marker.color = gray_point
            tr.fillcolor = gray_box_fill

        tr.marker.symbol = "circle"
        tr.marker.size = 4
        tr.line.color = "black"
        tr.line.width = 1
        tr.width = 0.75
        tr.jitter = 0.35
        tr.pointpos = 0.0

    fig.update_layout(
        boxmode="overlay",
        boxgap=0.28,
        height=800,
        width=800,
        legend_title_text=None,
        margin=dict(l=40, r=20, t=15, b=130),
        showlegend=color_boxes,
    )

    if horizontal:
        fig.update_xaxes(dtick=0.1)
    else:
        fig.update_yaxes(dtick=0.1)

    return fig


def plot_box_panel(
    df: pd.DataFrame,
    datasets: list[str],
    methods: list[str] = RM_METHODS + RCM_METHODS + MV_METHODS,
    reorder_by_median: bool = True,
    merge_mv_rcm: bool = True,
    color_boxes: bool = True,
    horizontal: bool = True,
    gray_box_fill: str = "rgba(180, 180, 180, 0.28)",
    gray_point: str = "rgba(90, 90, 90, 0.55)",
    acc_padding: float = 0.02,
    subtitle_fs: int = 14,
):
    # Filter to requested datasets and methods
    dff = df[df["dataset"].isin(datasets)].copy()
    if methods is not None:
        dff = dff[dff["method"].isin(methods)]

    if dff.empty:
        print(f"No entries found for datasets: {datasets}")
        return None

    if merge_mv_rcm:
        dff.loc[dff["mtype"] == "MV", "mtype"] = "RCM"

    # Determine method ordering
    if reorder_by_median:
        method_order = (
            dff.groupby("method")["acc"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        method_order = [m for m in methods if m in dff["method"].unique()]

    dff["method"] = pd.Categorical(dff["method"], categories=method_order, ordered=True)
    dff["dataset"] = pd.Categorical(dff["dataset"], categories=datasets, ordered=True)
    dff = dff.sort_values("method")

    x, y = ("acc", "method") if horizontal else ("method", "acc")

    fig = px.box(
        dff,
        x=x,
        y=y,
        color="mtype" if color_boxes else None,
        facet_col="dataset",
        points="all",
        category_orders={
            "method": method_order,
            "mtype": ["RCM", "MV", "RM"],
            "dataset": datasets,
        },
    )

    # Style the boxes
    STYLE = {
        "RCM": dict(marker="rgba(0, 60, 140, 0.5)", fill="rgba(0, 60, 140, 0.25)"),
        "MV": dict(marker="rgba(170, 90, 0, 0.5)", fill="rgba(170, 90, 0, 0.25)"),
        "RM": dict(marker="rgba(0, 0, 0, 0.3)", fill="rgba(0, 0, 0, 0.25)"),
    }

    for tr in fig.data:
        if color_boxes and (st := STYLE.get(tr.name)):
            tr.marker.color = st["marker"]
            tr.fillcolor = st["fill"]
        else:
            tr.marker.color = gray_point
            tr.fillcolor = gray_box_fill

        tr.marker.symbol = "circle"
        tr.marker.size = 4
        tr.line.color = "black"
        tr.line.width = 1
        tr.width = 0.75
        tr.jitter = 0.35
        tr.pointpos = 0.0

    # Layout
    n = len(datasets)
    acc_range = [-acc_padding, 1 + acc_padding]

    fig.update_layout(
        template="simple_white",
        boxmode="overlay",
        boxgap=0.28,
        height=800 if horizontal else 500,
        width=450 * n,
        legend=dict(
            title="",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        showlegend=color_boxes,
        margin=dict(l=120, r=80, t=50, b=50),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        showgrid=False,
        zeroline=False,
        mirror=True,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        ticks="outside",
        tickwidth=2,
        ticklen=8,
        showgrid=False,
        zeroline=False,
        mirror=True,
    )

    if horizontal:
        fig.update_xaxes(dtick=0.1)  # , range=acc_range)
    else:
        fig.update_yaxes(dtick=0.1)  # , range=acc_range)

    # Clean up facet titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_annotations(font_size=subtitle_fs)

    fig.update_yaxes(
        showticklabels=False,
        tickmode=None,
        tickvals=None,
        ticktext=None,
        showgrid=False,
        zeroline=False,
        ticks="",
        ticklen=0,
        row=1,
        col=2,
    )

    fig.update_yaxes(title="")

    return fig


""" --------------- heatmap plots ---------------"""


def plot_heatmap(
    df: pd.DataFrame,
    dataset: str,
    methods: list[str] = RCM_METHODS + RM_METHODS + MV_METHODS,
    reorder_by_mean: bool = True,
    show_text: bool = True,
    aggfunc: str = "median",
    zmin: float = 0,
    zmax: float = 1,
    colorscale: str = "Viridis",
    width: int = 600,
    height: int = 600,
):
    dff = df[df["dataset"] == dataset].copy()

    if dff.empty:
        print(f"No entries found for dataset: {dataset}")
        return None

    # Determine method ordering
    if reorder_by_mean:
        order = (
            dff.groupby("method")["acc"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = methods if methods else dff["method"].unique().tolist()

    # Filter to requested methods
    if methods is not None:
        order = [m for m in order if m in methods]
        dff = dff[dff["method"].isin(methods)]

    mat = dff.pivot_table(
        index="method", columns="task", values="acc", aggfunc=aggfunc
    ).reindex(index=order)

    z = mat.values
    text = None
    if show_text:
        text = np.where(
            np.isnan(z),
            "",
            np.vectorize(lambda x: f"{x:.2f}")(np.nan_to_num(z, nan=0.0)),
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=mat.columns.tolist(),
            y=mat.index.tolist(),
            text=text,
            texttemplate="%{text}" if show_text else None,
            textfont=dict(size=10),
            hovertemplate="method=%{y}<br>task=%{x}<br>acc=%{z:.4f}<extra></extra>",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=45, dtick=1)

    add_paper_styling(fig, lines=False)

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def plot_heatmap_panel(
    df: pd.DataFrame,
    datasets: list[str],
    methods: list[str] = RCM_METHODS + RM_METHODS + MV_METHODS,
    reorder_by_mean: bool = True,
    exclude_from_ordering: list[str] = None,
    show_text: bool = True,
    aggfunc: str = "median",
    zmin: float = 0,
    zmax: float = 1,
    colorscale: str = "Viridis",
    colorbar_title: str = "acc",
    min_col_width: float = 0.12,
    hspace: float = 0.03,
    height: int = 600,
    width: int = 1000,
    subtitle_fs: int = 14,
):
    # Determine method ordering across all datasets
    if reorder_by_mean:
        df_for_order = df[df["dataset"].isin(datasets)]
        if exclude_from_ordering:
            df_for_order = df_for_order[
                ~df_for_order["dataset"].isin(exclude_from_ordering)
            ]
        order = (
            df_for_order.groupby("method")["acc"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = methods if methods else df["method"].unique().tolist()

    # Filter to requested methods
    if methods is not None:
        order = [m for m in order if m in methods]

    # Build panels
    panels = []
    for ds in datasets:
        dff = df[(df["dataset"] == ds) & (df["method"].isin(order))].copy()

        mat = dff.pivot_table(
            index="method", columns="task", values="acc", aggfunc=aggfunc
        ).reindex(index=order)

        z = mat.values
        text = None
        if show_text:
            text = np.where(
                np.isnan(z),
                "",
                np.vectorize(lambda x: f"{x:.2f}")(np.nan_to_num(z, nan=0.0)),
            )

        trace = go.Heatmap(
            z=z,
            x=mat.columns.tolist(),
            y=mat.index.tolist(),
            text=text,
            texttemplate="%{text}" if show_text else None,
            textfont=dict(size=10),
            hovertemplate="method=%{y}<br>task=%{x}<br>acc=%{z:.4f}<extra></extra>",
            coloraxis="coloraxis",
        )

        col_width = max(mat.shape[1], 1)
        panels.append((trace, col_width))

    traces = [p[0] for p in panels]
    widths = [p[1] for p in panels]

    # Normalize column widths
    w = np.array(widths, float)
    w = w / w.sum()
    w = np.maximum(w, min_col_width)
    col_widths = (w / w.sum()).tolist()

    fig = make_subplots(
        rows=1,
        cols=len(traces),
        subplot_titles=datasets,
        column_widths=col_widths,
        horizontal_spacing=hspace,
    )

    for j, tr in enumerate(traces, start=1):
        fig.add_trace(tr, row=1, col=j)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=order,
            autorange="reversed",
            showticklabels=(j == 1),
            row=1,
            col=j,
        )

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(title=colorbar_title, len=0.9, y=0.5),
        )
    )

    add_paper_styling(fig, lines=False)

    for j in range(2, len(traces) + 1):
        fig.update_yaxes(
            showticklabels=False,
            ticks="",
            ticklen=0,
            row=1,
            col=j,
        )

    fig.update_xaxes(tickangle=45, dtick=1)
    fig.update_annotations(font_size=subtitle_fs)

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def plot_heatmap_with_marginals(
    df: pd.DataFrame,
    dataset: str,
    methods: list[str] = RCM_METHODS + MV_METHODS + RM_METHODS,
    reorder_by_mean: bool = True,
    show_text: bool = False,
    aggfunc: str = "median",
    zmin: float = 0,
    zmax: float = 1,
    colorscale: str = "Viridis",
    colorbar_title: str = "acc",
    height: int = 520,
    width: int = 900,
    marginal_fill: str = "rgba(180, 180, 180, 0.4)",
    marginal_line: str = "rgba(90, 90, 90, 0.8)",
):
    # Determine method ordering
    if reorder_by_mean:
        order = (
            df[df["dataset"] == dataset]
            .groupby("method")["acc"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = methods if methods else df["method"].unique().tolist()

    if methods is not None:
        order = [m for m in order if m in methods]

    dff = df[(df["dataset"] == dataset) & (df["method"].isin(order))].copy()

    if dff.empty:
        print(f"No entries found for dataset: {dataset}")
        return None

    # Order tasks grouped by ref, preserve appearance within each ref
    tmp = dff[["task", "ref"]].drop_duplicates("task").copy()
    tmp["ord"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["ref", "ord"], kind="stable")
    tasks = tmp["task"].tolist()
    refs = tmp["ref"].tolist()

    # Contiguous groups after sorting by ref
    groups = []
    start = 0
    for i in range(1, len(tasks) + 1):
        if i == len(tasks) or refs[i] != refs[i - 1]:
            groups.append((refs[start], start, i - 1))
            start = i

    # Matrix
    mat = dff.pivot_table(
        index="method", columns="task", values="acc", aggfunc=aggfunc
    ).reindex(index=order, columns=tasks)

    long = (
        mat.reset_index()
        .melt(id_vars="method", var_name="task", value_name="acc")
        .dropna()
    )

    # Subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.84, 0.16],
        row_heights=[0.82, 0.18],
        specs=[
            [{"type": "heatmap"}, {"type": "box"}],
            [{"type": "box"}, {"type": "xy"}],
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Heatmap
    z = mat.values
    text = None
    if show_text:
        text = np.where(
            np.isnan(z),
            "",
            np.vectorize(lambda x: f"{x:.2f}")(np.nan_to_num(z, nan=0.0)),
        )

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=tasks,
            y=order,
            text=text,
            texttemplate="%{text}" if show_text else None,
            textfont=dict(size=10),
            hovertemplate="method=%{y}<br>task=%{x}<br>acc=%{z:.4f}<extra></extra>",
            coloraxis="coloraxis",
        ),
        row=1,
        col=1,
    )

    # Right marginal (per-method over tasks)
    fig.add_trace(
        go.Box(
            y=long["method"],
            x=long["acc"],
            orientation="h",
            boxpoints=False,
            line=dict(width=1, color=marginal_line),
            fillcolor=marginal_fill,
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Bottom marginal (per-task over methods)
    long_b = long.copy()
    long_b["task"] = pd.Categorical(long_b["task"], categories=tasks, ordered=True)

    fig.add_trace(
        go.Box(
            x=long_b["task"],
            y=long_b["acc"],
            boxpoints=False,
            line=dict(width=1, color=marginal_line),
            fillcolor=marginal_fill,
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    # Shared coloraxis
    fig.update_layout(
        coloraxis=dict(
            cmin=zmin,
            cmax=zmax,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, len=0.78, y=0.58),
        ),
        template="simple_white",
        margin=dict(l=10, r=10, t=65, b=55),
        height=height,
        width=width,
    )

    # Enforce method ordering
    fig.update_yaxes(
        categoryorder="array", categoryarray=order, autorange="reversed", row=1, col=1
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=order, autorange="reversed", row=1, col=2
    )

    # Header bands + labels above heatmap, and vertical dividers at ref boundaries
    for ref_name, s, e in groups:
        if s > 0:
            fig.add_shape(
                type="line",
                x0=s - 0.5,
                x1=s - 0.5,
                y0=-0.5,
                y1=len(order) - 0.5,
                xref="x",
                yref="y",
                line=dict(width=2, color="white"),
                layer="above",
            )

        fig.add_shape(
            type="rect",
            x0=s - 0.5,
            x1=e + 0.5,
            y0=1.02,
            y1=1.12,
            xref="x",
            yref="y domain",
            line=dict(width=0),
            fillcolor="rgba(240,240,240,1.0)",
            layer="above",
        )

        cx = (s + e) / 2.0
        fig.add_annotation(
            x=cx,
            y=1.07,
            xref="x",
            yref="y domain",
            text=f"<b>{ref_name}</b>",
            showarrow=False,
            font=dict(size=11),
        )

    # Apply paper styling to heatmap axes only
    add_paper_styling(fig, lines=False)

    # Marginals axis styling (thinner, subtler)
    eps = 0.05
    # Shared border styling for marginals
    marginal_border_style = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=False,
        zeroline=False,
        mirror=True,
    )

    # Bottom marginal
    fig.update_xaxes(
        showticklabels=False,
        ticks="",
        ticklen=0,
        **marginal_border_style,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[-eps, 1 + eps],
        fixedrange=True,
        tickmode="array",
        tickvals=[0, 0.5, 1.0],
        ticktext=["0", "0.5", "1"],
        ticks="outside",
        ticklen=3,
        tickwidth=1,
        title_text="",
        **marginal_border_style,
        row=2,
        col=1,
    )

    # Right marginal
    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        ticklen=0,
        **marginal_border_style,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        range=[-eps, 1 + eps],
        fixedrange=True,
        tickmode="array",
        tickvals=[0, 0.5, 1.0],
        ticktext=["0", "0.5", "1"],
        ticks="outside",
        ticklen=3,
        tickwidth=1,
        title_text="",
        **marginal_border_style,
        row=1,
        col=2,
    )

    # Hide heatmap x-axis labels
    fig.update_xaxes(
        showticklabels=False,
        ticks="",
        ticklen=0,
        row=1,
        col=1,
    )

    # Dataset title at bottom
    fig.add_annotation(
        x=0.5,
        y=-0.24,
        xref="paper",
        yref="paper",
        text=f"<b>{dataset}</b>",
        showarrow=False,
        font=dict(size=13),
    )

    # Ensure shapes are white dividers
    fig.update_shapes(line=dict(color="white", width=2), layer="above")

    return fig


def plot_cfmatrix(
    annot: Annot,
    title: str = "Confusion Matrix",
    x_label: str = "Predicted (Reference)",
    y_label: str = "True (Query)",
    show_all_labels: bool = True,
    show_values: bool = False,
    normalize: bool = False,
    colorscale: str = "agsunset_r",
    angle_x_labels: bool = False,
    width: float = -1,
    height: float = -1,
):
    """
    Display confusion matrix heatmap from an Annot object.

    Parameters
    ----------
    annot : Annot
        Annot object with evaluated confusion matrix
    title : str
        Plot title
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    show_all_labels : bool
        Whether to show all tick labels
    show_values : bool
        Whether to show values in cells
    normalize : bool
        Whether to normalize rows to sum to 1
    colorscale : str
        Plotly colorscale name
    angle_x_labels : bool
        Whether to angle x labels at 45 deg instead of 90 deg
    width : float
        Figure width (if positive)
    height : float
        Figure height (if positive)

    Returns
    -------
    go.Figure
    """
    if annot._cfmatrix is None:
        raise ValueError("Confusion matrix not computed. Run annot.eval_() first.")

    cfmatrix = (
        annot._cfmatrix[:, :-1].copy()
        if annot._rlabels[-1] == "novel"
        else annot._cfmatrix.copy()
    )

    z = cfmatrix.astype(float)
    if normalize:
        row_sums = z.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        z = z / row_sums

    n_rows, n_cols = z.shape
    q_labels = annot._qlabels
    r_labels = annot._rlabels

    # Build hovertext
    hovertext = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            val = z[i, j]
            row.append(
                f"{y_label}: {q_labels[i]}<br>"
                f"{x_label}: {r_labels[j]}<br>"
                f"{'proportion' if normalize else 'count'}: {val:.4f}"
            )
        hovertext.append(row)

    # Create heatmap
    heatmap = go.Heatmap(
        z=z,
        x=list(range(n_cols)),
        y=list(range(n_rows)),
        colorscale=colorscale,
        colorbar=dict(title="proportion" if normalize else "count"),
        hoverinfo="text",
        text=hovertext,
        xgap=0,
        ygap=0,
        zmin=0,
        zmax=1 if normalize else z.max(),
    )

    fig = go.Figure(data=[heatmap])

    # Configure axes
    fig.update_xaxes(
        title=x_label,
        tickmode="array",
        tickvals=list(range(n_cols)),
        ticktext=r_labels,
        tickangle=-90 if not angle_x_labels else -45,
        range=[-0.5, n_cols - 0.5],
        constrain="domain",
    )
    fig.update_yaxes(
        title=y_label,
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=q_labels,
        range=[n_rows - 0.5, -0.5],  # Reversed for top-to-bottom
        scaleanchor="x",
        constrain="domain",
    )

    fig.update_layout(title=title)

    # Show values as annotations
    if show_values:
        for i in range(n_rows):
            for j in range(n_cols):
                val = z[i, j]
                text = f"{val:.2f}" if normalize else f"{int(cfmatrix[i, j])}"
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(color="white"),
                )

    # Final layout tweaks
    if width > 0:
        fig.update_layout(width=width)
    if height > 0:
        fig.update_layout(height=height)
    if show_all_labels:
        fig.update_yaxes(dtick=1)
        fig.update_xaxes(dtick=1)

    add_paper_styling(fig, lines=False)

    return fig
