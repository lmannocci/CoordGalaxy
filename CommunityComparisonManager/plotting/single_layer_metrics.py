"""Single-layer community metric visualizations for overlap analysis."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from utils.common_variables import co_action_column_print, co_action_column_print3, color_dict2, dpi, dtype
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class SingleLayerMetricsPlotter:
    """Plot comparisons of structural metrics for matched single-layer communities."""

    def _get_offsets_russia1(self) -> dict[tuple[str, int, int], tuple[float, float]]:
        """
        Return custom scatter-label offsets for the Russia1 plots.

        :return: [dict] Mapping from (layer, layer community, generic community) to x/y offset.
        """
        if self.file_prefix == "louvain_resolution_1":
            return {
                ("co-mention", 0, 1): (15, 0),
                ("co-mention", 1, 2): (0, -10),
                ("co-mention", 3, 0): (0, -10),
                ("co-mention", 5, 4): (15, 0),
                ("co-mention", 6, 5): (15, 0),
                ("co-hashtag", 0, 2): (10, 5),
                ("co-hashtag", 1, 1): (0, -10),
                ("co-hashtag", 3, 4): (15, 0),
                ("co-hashtag", 4, 0): (-15, 0),
                ("co-URL", 0, 1): (10, 0),
                ("co-URL", 1, 0): (0, 8),
            }
        if self.file_prefix == "infomap":
            return {
                ("co-mention", 0, 0): (0, -2),
                ("co-mention", 2, 3): (-0.5, -2),
                ("co-hashtag", 0, 0): (0, -2),
                ("co-hashtag", 2, 1): (100, 0),
                ("co-URL", 0, 1): (100, 0),
                ("co-URL", 2, 4): (0, -2),
            }
        return {}

    def _get_offsets_uk(self) -> dict[tuple[str, int, int], tuple[float, float]]:
        """
        Return custom scatter-label offsets for the UK plots.

        :return: [dict] Mapping from (layer, layer community, generic community) to x/y offset.
        """
        if self.file_prefix == "louvain_resolution_1":
            return {
                ("co-mention", 0, 0): (0, 25),
                ("co-mention", 1, 1): (0, 25),
                ("co-mention", 2, 2): (500, 0),
                ("co-mention", 3, 3): (-25, 25),
                ("co-mention", 4, 4): (500, 0),
                ("co-hashtag", 0, 0): (0, 25),
                ("co-hashtag", 2, 1): (500, -20),
                ("co-hashtag", 3, 2): (-400, 28),
                ("co-hashtag", 5, 4): (50, 30),
                ("co-URL", 4, 1): (400, -20),
            }
        if self.file_prefix == "infomap":
            return {
                ("co-mention", 0, 0): (-1000, 0),
                ("co-mention", 1, 1): (0, 20),
                ("co-mention", 5, 2): (0, 20),
                ("co-hashtag", 0, 0): (-1000, 0),
                ("co-URL", 0, 0): (400, -20),
            }
        return {}

    def _get_offsets(self, point: pd.Series) -> tuple[float, float]:
        """
        Get the custom text offset for one scatter point.

        :param point: [pd.Series] Row containing layer, com_layer, and com_generic.
        :return: [tuple[float, float]] X/Y label offset.
        """
        if self.dataset_name == "uk":
            offset_conditions = self._get_offsets_uk()
        elif self.dataset_name == "russia1":
            offset_conditions = self._get_offsets_russia1()
        else:
            offset_conditions = {}

        updated_offset_conditions = {
            (co_action_column_print3.get(key[0], key[0]), key[1], key[2]): value
            for key, value in offset_conditions.items()
        }
        key = (point["layer"], point["com_layer"], point["com_generic"])
        return updated_offset_conditions.get(key, (0, 0))

    def _load_single_layer_metric_inputs(
        self,
        mid_th: float,
        metric: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and normalize the input dataframes used by single-layer metric plots.

        :param mid_th: [float] Flux threshold encoded in the filename.
        :param metric: [str] Overlap metric encoded in the filename.
        :return: [tuple[pd.DataFrame, pd.DataFrame]] Flux dataframe and metrics dataframe.
        """
        flux_df = self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_"
            f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype,
        )
        metrics_df = self.ch.read_dataframe(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_single_layer_metrics_communities.csv",
            dtype=dtype,
        )
        flux_df["layer"] = flux_df["layer"].replace(co_action_column_print)
        flux_df["generic"] = flux_df["generic"].replace(co_action_column_print)
        metrics_df["layer"] = metrics_df["layer"].replace(co_action_column_print)
        return flux_df, metrics_df

    def _merge_common_metrics(self, metrics_df: pd.DataFrame, common_df: pd.DataFrame, generic: str) -> pd.DataFrame:
        """
        Merge common-community flux rows with their structural metrics.

        :param metrics_df: [pd.DataFrame] Community metrics for all layers.
        :param common_df: [pd.DataFrame] Flux rows labelled as common.
        :param generic: [str] Generic layer currently being compared.
        :return: [pd.DataFrame] Metrics for matched layer/generic community pairs.
        """
        plot_df1 = common_df[common_df["layer"] != generic].merge(metrics_df, on=["layer", "com_layer"], how="inner")
        plot_df2 = common_df[common_df["layer"] == generic].merge(
            plot_df1[["generic", "com_generic"]].drop_duplicates(),
            on=["generic", "com_generic"],
            how="inner",
        )
        plot_df2 = plot_df2.merge(metrics_df, on=["layer", "com_layer"], how="inner")
        return pd.concat([plot_df1, plot_df2], ignore_index=True)

    def _get_perplexity(self, n_rows: int) -> int:
        """
        Choose a t-SNE perplexity compatible with the dataframe size.

        :param n_rows: [int] Number of rows to embed.
        :return: [int] Perplexity value.
        """
        return 10 if n_rows > 10 else max(n_rows - 1, 1)

    def _get_n_neighbors(self, n_rows: int) -> int:
        """
        Choose a UMAP neighbour count compatible with the dataframe size.

        :param n_rows: [int] Number of rows to embed.
        :return: [int] UMAP n_neighbors value.
        """
        return 10 if n_rows > 10 else max(n_rows - 1, 1)

    def _add_global_embedding(
        self,
        metrics_df: pd.DataFrame,
        visualization: str,
        features: list[str],
    ) -> tuple[pd.DataFrame, str | None, str | None, int | None]:
        """
        Compute the global 2D embedding used as a reference in scatter plots.

        :param metrics_df: [pd.DataFrame] Metrics dataframe.
        :param visualization: [str] Visualization type: t_sne, umap, pca, starplot, or cosine_similarity.
        :param features: [list[str]] Metric columns used for the embedding.
        :return: [tuple] Updated dataframe, x column, y column, and global model parameter.
        """
        if visualization == "t_sne":
            perplexity = self._get_perplexity(metrics_df.shape[0])
            self.lm.printl(f"{file_name}. global t-SNE perplexity: {perplexity}")
            embedding = TSNE(n_components=2, perplexity=perplexity, random_state=0, n_jobs=1).fit_transform(
                metrics_df[features]
            )
            metrics_df[["x_tsne_all", "y_tsne_all"]] = embedding
            return metrics_df, "x_tsne_all", "y_tsne_all", perplexity
        if visualization == "umap":
            n_neighbors = self._get_n_neighbors(metrics_df.shape[0])
            embedding = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=2,
                metric="manhattan",
                random_state=42,
            ).fit_transform(metrics_df[features])
            metrics_df[["x_umap_all", "y_umap_all"]] = embedding
            return metrics_df, "x_umap_all", "y_umap_all", n_neighbors
        if visualization == "pca":
            embedding = PCA(n_components=2).fit_transform(metrics_df[features])
            metrics_df[["x_pca_all", "y_pca_all"]] = embedding
            return metrics_df, "x_pca_all", "y_pca_all", None
        return metrics_df, None, None, None

    def _scatterplot(
        self,
        features: list[str],
        plot_df: pd.DataFrame,
        x: str,
        y: str,
        generic: str,
        title: str,
        filename: str,
    ) -> None:
        """
        Plot a 2D embedding of matched communities.

        :param features: [list[str]] Structural metric columns used for the embedding.
        :param plot_df: [pd.DataFrame] Matched-community dataframe.
        :param x: [str] X-axis embedding column.
        :param y: [str] Y-axis embedding column.
        :param generic: [str] Generic layer label.
        :param title: [str] Figure title.
        :param filename: [str] Output image path.
        :return: None. The figure is saved to disk.
        """
        self.lm.printl(f"{file_name}. _scatterplot size: {str(plot_df.shape)}.")
        unique_com_generic = plot_df["com_generic"].dropna().unique()
        colors = sns.color_palette("pastel", len(unique_com_generic))
        color_map = dict(zip(unique_com_generic, colors))
        unique_layers = plot_df["layer"].unique()
        markers = ["o", "s", "d", "^", "*"][0: len(unique_layers)]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x=x,
            y=y,
            hue="com_generic",
            style="layer",
            palette=color_map,
            s=150,
            markers=markers,
            legend=False,
            edgecolor="white",
            linewidth=1.2,
        )

        layer_handles = [
            Line2D([0], [0], marker=markers[i], color="w", markerfacecolor="gray", markersize=10)
            for i in range(len(unique_layers))
        ]
        if self.dataset_name == "russia1" and self.file_prefix == "louvain_resolution_1":
            legend_layer = plt.legend(layer_handles, unique_layers, title="Layer", loc="best", borderaxespad=0.5)
        else:
            legend_layer = plt.legend(
                layer_handles,
                unique_layers,
                title="Layer",
                loc="best",
                bbox_to_anchor=(1, 1),
                borderaxespad=0.5,
            )

        labels_legend = ["group " + str(item) for item in list(color_map.keys())]
        handles_generic = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[cat], markersize=10)
            for cat in color_map
        ]
        legend_generic = plt.legend(
            handles=handles_generic,
            labels=labels_legend,
            title="Community RTW",
            loc="best",
            bbox_to_anchor=(1, 0.7),
            borderaxespad=0.5,
        )
        plt.gca().add_artist(legend_layer)
        plt.gca().add_artist(legend_generic)
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        for _, point_a in plot_df[plot_df["layer"] != generic].iterrows():
            point_b = plot_df[(plot_df["com_generic"] == point_a["com_generic"]) & (plot_df["layer"] == generic)]
            if point_b.empty:
                continue
            point_b = point_b.iloc[0]
            line_color = color_map[point_a["com_generic"]]
            plt.plot([point_a[x], point_b[x]], [point_a[y], point_b[y]], color=line_color, linewidth=2)
            offset_x, offset_y = self._get_offsets(point_a)
            plt.text(
                point_a[x] + offset_x,
                point_a[y] + offset_y,
                str(point_a["layer"]),
                color=line_color,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.xlabel("")
        plt.ylabel("")
        ax = plt.gca()
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        try:
            plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
        except Exception as error:
            self.lm.printl(error)
            plt.savefig(filename, dpi=dpi)
        plt.show()

    def _plot_embedding_for_generic(
        self,
        visualization: str,
        plot_df: pd.DataFrame,
        features: list[str],
        generic: str,
        x_all: str,
        y_all: str,
        global_parameter: int | None,
        metric: str,
        mid_th: float,
    ) -> None:
        """
        Compute and plot the local embedding for one generic layer.

        :param visualization: [str] Embedding type: t_sne, umap, or pca.
        :param plot_df: [pd.DataFrame] Matched-community dataframe.
        :param features: [list[str]] Metric columns used for the embedding.
        :param generic: [str] Generic layer label.
        :param x_all: [str] Global embedding x column.
        :param y_all: [str] Global embedding y column.
        :param global_parameter: [int | None] Global t-SNE perplexity or UMAP neighbours.
        :param metric: [str] Overlap metric name.
        :param mid_th: [float] Flux threshold.
        :return: None. Figures are saved to disk.
        """
        if visualization == "t_sne":
            local_parameter = self._get_perplexity(plot_df.shape[0])
            embedding = TSNE(n_components=2, perplexity=local_parameter, random_state=0, n_jobs=1).fit_transform(
                plot_df[features]
            )
            x_generic, y_generic = "x_tsne_generic", "y_tsne_generic"
            suffix = (
                f"{self.dm.path_overlapping_t_sne_plot}{self.file_prefix}_{generic}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}"
            )
            filename_all = f"{suffix}_tsne_embedding_all_perplexity_{str(global_parameter)}.png"
            filename_generic = f"{suffix}_tsne_embedding_generic_perplexity_{str(local_parameter)}.png"
            title = f"t-SNE Visualization of common communities of {generic}"
        elif visualization == "umap":
            local_parameter = self._get_n_neighbors(plot_df.shape[0])
            suffix = (
                f"{self.dm.path_overlapping_umap_plot}{self.file_prefix}_{generic}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}"
            )
            filename_all = f"{suffix}_umap_embedding_all_neighbors_{str(global_parameter)}.png"
            title = f"UMAP Visualization of common communities of {generic}"
            if plot_df.shape[0] <= 2:
                self.lm.printl(f"{file_name}. Generic {generic}: not enough data points for local UMAP.")
                self._scatterplot(features, plot_df, x_all, y_all, generic, title, filename_all)
                return
            embedding = umap.UMAP(
                n_neighbors=local_parameter,
                n_components=2,
                metric="manhattan",
                random_state=42,
            ).fit_transform(plot_df[features])
            x_generic, y_generic = "x_umap_generic", "y_umap_generic"
            filename_generic = f"{suffix}_umap_embedding_generic_neighbors_{str(local_parameter)}.png"
        else:
            embedding = PCA(n_components=2).fit_transform(plot_df[features])
            x_generic, y_generic = "x_pca_generic", "y_pca_generic"
            suffix = (
                f"{self.dm.path_overlapping_pca_plot}{self.file_prefix}_{generic}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}"
            )
            filename_all = f"{suffix}_pca_embedding_all.png"
            filename_generic = f"{suffix}_pca_embedding_generic.png"
            title = f"PCA Visualization of common communities of {generic}"

        plot_df[[x_generic, y_generic]] = embedding
        self._scatterplot(features, plot_df, x_all, y_all, generic, title, filename_all)
        self._scatterplot(features, plot_df, x_generic, y_generic, generic, title, filename_generic)

    def _plot_starplot(
        self,
        starplot_df: pd.DataFrame,
        features: list[str],
        normalized: bool,
        metric: str | None = None,
        mid_th: float | None = None,
        ax=None,
    ) -> None:
        """
        Plot one radar chart comparing a layer community with its generic match.

        :param starplot_df: [pd.DataFrame] Two-row dataframe: layer row and generic row.
        :param features: [list[str]] Metric columns to plot.
        :param normalized: [bool] Whether normalized metric columns should be used.
        :param metric: [str | None] Overlap metric name used for filenames.
        :param mid_th: [float | None] Flux threshold used for filenames.
        :param ax: [matplotlib.axes.Axes | None] Optional polar axis.
        :return: None. The figure is saved when no axis is passed.
        """
        features_to_plot = ["norm_" + f for f in features] if normalized else features
        generic = starplot_df.iloc[0]["generic"]
        layer = starplot_df.iloc[0]["layer"]
        values_layer = starplot_df.iloc[0][features_to_plot].values.tolist()
        values_generic = starplot_df.iloc[1][features_to_plot].values.tolist()
        com_layer = str(int(starplot_df.iloc[0]["com_layer"]))
        com_generic = str(int(starplot_df.iloc[0]["com_generic"]))

        angles = np.linspace(0, 2 * np.pi, len(features_to_plot), endpoint=False).tolist()
        angles += angles[:1]
        values_layer += values_layer[:1]
        values_generic += values_generic[:1]

        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            show_plot = True
        else:
            show_plot = False

        ax.fill(angles, values_layer, color=color_dict2[layer], alpha=0.5, label=f"{layer}")
        ax.fill(angles, values_generic, color=color_dict2[generic], alpha=0.5, label=f"{generic}")
        ax.plot(angles, values_layer, color=color_dict2[layer], linewidth=2)
        ax.plot(angles, values_generic, color=color_dict2[generic], linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, fontsize=16)
        ax.grid(color="gray", linestyle="--", linewidth=1.5)
        ax.spines["polar"].set_visible(False)
        for radius in np.linspace(0.2, 1, 5):
            ax.text(
                np.pi / 2 - (1 / radius) * 0.22,
                radius + 0.09,
                f"{radius:.2f}",
                color="gray",
                ha="center",
                va="center",
                fontsize=12,
            )
        ax.set_ylim(0, 1)
        ax.set_title(f"{generic}_{com_generic} and {layer}_{com_layer}", fontsize=18, ha="center", va="bottom", fontweight="bold")

        if show_plot:
            plt.tight_layout()
            filename = (
                f"{self.dm.path_overlapping_starplot}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_"
                f"{generic}_{layer}_{com_generic}_{com_layer}_starplot.png"
            )
            plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.show()

    def _plot_legend_starplot(self, plot_df: pd.DataFrame, generic: str) -> None:
        """
        Save a standalone legend for starplots.

        :param plot_df: [pd.DataFrame] Matched-community dataframe.
        :param generic: [str] Generic layer label.
        :return: None. The legend image is saved to disk.
        """
        if plot_df.empty:
            return
        list_show_layer = list(set(plot_df["layer"].unique()).union(set(plot_df["generic"].unique())))
        filtered_color_dict = {key: value for key, value in color_dict2.items() if key in list_show_layer}
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        handles = [plt.Line2D([0], [0], color=color, lw=6) for color in filtered_color_dict.values()]
        labels = list(filtered_color_dict.keys())
        ax.legend(handles, labels, loc="center", frameon=False, ncol=len(filtered_color_dict), bbox_to_anchor=(0.5, 0.5), bbox_transform=fig.transFigure)
        plt.savefig(f"{self.dm.path_overlapping_starplot}{generic}_starplot_legend.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.show()

    def _plot_grid_starplots(
        self,
        plot_df: pd.DataFrame,
        generic: str,
        features: list[str],
        normalized: bool,
        mid_th: float,
        metric: str,
    ) -> None:
        """
        Plot radar charts in one grid per layer.

        :param plot_df: [pd.DataFrame] Matched-community dataframe.
        :param generic: [str] Generic layer label.
        :param features: [list[str]] Metric columns to plot.
        :param normalized: [bool] Whether normalized metric columns should be used.
        :param mid_th: [float] Flux threshold used for filenames.
        :param metric: [str] Overlap metric used for filenames.
        :return: None. The grid figures are saved to disk.
        """
        for layer in plot_df[plot_df["layer"] != generic]["layer"].unique():
            layer_rows = plot_df[plot_df["layer"] == layer]
            num_starplots = len(layer_rows)
            ncols = min(num_starplots, 3)
            nrows = (num_starplots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), subplot_kw=dict(polar=True))
            axes = np.array([axes]) if nrows == 1 and ncols == 1 else axes.flatten()
            plt.suptitle(f"Generic: {generic} - Layer {layer}", fontsize=15, fontweight="bold", y=1.05)

            plot_index = 0
            for _, row in layer_rows.iterrows():
                if plot_index >= len(axes):
                    break
                row2 = plot_df.loc[(plot_df["layer"] == generic) & (plot_df["com_generic"] == row["com_generic"])]
                starplot_df = pd.concat([pd.DataFrame([row]), row2], ignore_index=True)
                self._plot_starplot(starplot_df, features, normalized, ax=axes[plot_index])
                plot_index += 1

            for i in range(plot_index, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            filename = (
                f"{self.dm.path_overlapping_starplot}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_{generic}_{layer}_grid_starplot.png"
            )
            plt.savefig(filename, dpi=800, bbox_inches="tight", pad_inches=0)
            plt.show()

    def _plot_starplots_for_generic(
        self,
        metrics_df: pd.DataFrame,
        common_df: pd.DataFrame,
        generic: str,
        features: list[str],
        normalized: bool,
        type_visualization_starplot: str,
        mid_th: float,
        metric: str,
    ) -> None:
        """
        Plot starplots for one generic layer.

        :param metrics_df: [pd.DataFrame] Metrics dataframe.
        :param common_df: [pd.DataFrame] Flux rows labelled as common.
        :param generic: [str] Generic layer label.
        :param features: [list[str]] Metric columns to plot.
        :param normalized: [bool] Whether normalized metric columns should be used.
        :param type_visualization_starplot: [str] Starplot layout: single or grid.
        :param mid_th: [float] Flux threshold used for filenames.
        :param metric: [str] Overlap metric used for filenames.
        :return: None. Figures are saved to disk.
        """
        if normalized:
            scaler = MinMaxScaler()
            metrics_df[["norm_" + feature for feature in features]] = scaler.fit_transform(metrics_df[features])

        plot_df = self._merge_common_metrics(metrics_df, common_df, generic)
        if plot_df.empty:
            self.lm.printl(f"{file_name}. Empty dataframe for {generic}. No starplot generated.")
            return
        self._plot_legend_starplot(plot_df, generic)

        if type_visualization_starplot == "single":
            for _, row in plot_df[plot_df["layer"] != generic].iterrows():
                row2 = plot_df.loc[(plot_df["layer"] == generic) & (plot_df["com_generic"] == row["com_generic"])]
                starplot_df = pd.concat([pd.DataFrame([row]), row2], ignore_index=True)
                self._plot_starplot(starplot_df, features, normalized, metric=metric, mid_th=mid_th)
        elif type_visualization_starplot == "grid":
            self._plot_grid_starplots(plot_df, generic, features, normalized, mid_th, metric)

    def _compute_cosine_similarity_for_generic(
        self,
        metrics_df: pd.DataFrame,
        common_df: pd.DataFrame,
        generic: str,
        features: list[str],
        normalized: bool,
    ) -> list[dict[str, object]]:
        """
        Compute cosine similarity between matched layer/generic community metrics.

        :param metrics_df: [pd.DataFrame] Metrics dataframe.
        :param common_df: [pd.DataFrame] Flux rows labelled as common.
        :param generic: [str] Generic layer label.
        :param features: [list[str]] Metric columns to compare.
        :param normalized: [bool] Whether normalized metric columns should be used.
        :return: [list[dict[str, object]]] Cosine similarity rows.
        """
        if normalized:
            scaler = MinMaxScaler()
            selected_features = ["norm_" + feature for feature in features]
            metrics_df[selected_features] = scaler.fit_transform(metrics_df[features])
        else:
            selected_features = features

        results = []
        plot_df = self._merge_common_metrics(metrics_df, common_df, generic)
        for _, row in plot_df[plot_df["layer"] != generic].iterrows():
            row2 = plot_df.loc[(plot_df["layer"] == generic) & (plot_df["com_generic"] == row["com_generic"])]
            starplot_df = pd.concat([pd.DataFrame([row]), row2], ignore_index=True)
            vector1 = starplot_df[selected_features].values[0].reshape(1, -1)
            vector2 = starplot_df[selected_features].values[1].reshape(1, -1)
            results.append(
                {
                    "com_layer": str(int(starplot_df.iloc[0]["com_layer"])),
                    "com_generic": str(int(starplot_df.iloc[0]["com_generic"])),
                    "generic": generic,
                    "layer": starplot_df.iloc[0]["layer"],
                    "cosine_similarity": cosine_similarity(vector1, vector2)[0][0],
                }
            )
        return results

    def _darken(self, color: Iterable[float], factor: float = 0.7) -> tuple[float, ...]:
        """
        Darken an RGB color by multiplying each channel.

        :param color: [Iterable[float]] RGB color.
        :param factor: [float] Multiplicative factor between 0 and 1.
        :return: [tuple[float, ...]] Darkened RGB color.
        """
        return tuple(channel * factor for channel in color)

    def _get_offset_cosine_similarity(self, generic: str, layer: str) -> float:
        """
        Return the custom annotation offset for cosine-similarity mean labels.

        :param generic: [str] Generic layer label.
        :param layer: [str] Compared layer label.
        :return: [float] X-axis label offset.
        """
        if generic != "RTW":
            return 0
        if self.dataset_name == "uk" and self.file_prefix == "louvain_resolution_1" and layer == "HST":
            return -0.4
        if self.dataset_name == "russia1" and self.file_prefix == "louvain_resolution_1":
            if layer == "MEN":
                return -1
            if layer == "HST":
                return -0.4
        if self.dataset_name == "russia1" and self.file_prefix == "infomap" and layer in {"URL", "HST"}:
            return -0.2
        return 0

    @log_method
    def plot_single_layer_metrics(
        self,
        visualization: str,
        type_visualization_starplot: str = "single",
        normalized: bool = True,
        features: list[str] | None = None,
        mid_th: float = 0.5,
        metric: str = "harmonicMean",
    ) -> None:
        """
        Plot or compute metric comparisons for common single-layer communities.

        :param visualization: [str] One of t_sne, umap, pca, starplot, cosine_similarity.
        :param type_visualization_starplot: [str] Starplot layout: single or grid.
        :param normalized: [bool] Whether to normalize metric vectors before starplot/cosine operations.
        :param features: [list[str] | None] Metric columns to use.
        :param mid_th: [float] Flux threshold encoded in input/output filenames.
        :param metric: [str] Overlap metric encoded in input/output filenames.
        :return: None. Figures or cosine-similarity CSVs are saved to disk.
        """
        features = features or ["size", "density", "avg_weight", "conductance", "avg_degree", "avg_clustering", "assortativity"]
        flux_df, metrics_df = self._load_single_layer_metric_inputs(mid_th, metric)
        metrics_df, x_all, y_all, global_parameter = self._add_global_embedding(metrics_df, visualization, features)

        cosine_results = []
        for generic in metrics_df["layer"].unique():
            common_df = flux_df.loc[
                (flux_df["label"] == "common")
                & (flux_df["generic"] == generic)
                & (flux_df["layer"] != "generic")
            ].copy()
            common_df["com_layer"] = common_df["com_layer"].astype("int")
            common_df["com_generic"] = common_df["com_generic"].astype("int")

            if visualization in {"t_sne", "umap", "pca"}:
                plot_df = self._merge_common_metrics(metrics_df, common_df, generic)
                if plot_df.empty:
                    self.lm.printl(f"{file_name}. Empty dataframe for {generic}. No embedding plot generated.")
                    continue
                self._plot_embedding_for_generic(
                    visualization,
                    plot_df,
                    features,
                    generic,
                    x_all,
                    y_all,
                    global_parameter,
                    metric,
                    mid_th,
                )
            elif visualization == "starplot":
                self._plot_starplots_for_generic(
                    metrics_df,
                    common_df,
                    generic,
                    features,
                    normalized,
                    type_visualization_starplot,
                    mid_th,
                    metric,
                )
            elif visualization == "cosine_similarity":
                cosine_results.extend(
                    self._compute_cosine_similarity_for_generic(metrics_df, common_df, generic, features, normalized)
                )
            else:
                raise ValueError(f"Unsupported single-layer metrics visualization: {visualization}")

        if visualization == "cosine_similarity":
            cosine_results_df = pd.DataFrame(cosine_results)
            self.ch.save_dataframe(
                cosine_results_df,
                f"{self.dm.path_cosine_similarity}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_"
                f"normalized_{str(normalized)}_single_layer_metrics_cosine_similarity.csv",
            )

    @log_method
    def plot_barchart_cosine_similarity(
        self,
        generic: str = "co-retweet",
        normalized: bool = True,
        metric: str = "harmonicMean",
        mid_th: float = 0.5,
    ) -> None:
        """
        Plot sorted cosine-similarity bars for one generic layer.

        :param generic: [str] Generic layer to plot. Full co-action names are converted to plot labels.
        :param normalized: [bool] Whether the source cosine file used normalized metrics.
        :param metric: [str] Overlap metric encoded in the source filename.
        :param mid_th: [float] Flux threshold encoded in the source filename.
        :return: None. The bar chart is saved to disk.
        """
        cosine_df = self.ch.read_dataframe(
            f"{self.dm.path_cosine_similarity}{self.file_prefix}_{metric}_"
            f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_"
            f"normalized_{str(normalized)}_single_layer_metrics_cosine_similarity.csv",
            dtype=dtype,
        )
        generic_label = co_action_column_print.get(generic, generic)
        df = cosine_df[cosine_df["generic"] == generic_label]
        dark_palette = {key: self._darken(value, 0.7) for key, value in color_dict2.items()}

        plt.figure(figsize=(9, 6), dpi=300)
        xpos = []
        colors = []
        values = []
        counter = 0
        block_centers = {}
        layer_order = ["RPL", "URL", "MEN", "HST"]
        ordered_layers = [layer for layer in layer_order if layer in df["layer"].unique()]

        for layer in ordered_layers:
            vals = np.sort(df[df["layer"] == layer]["cosine_similarity"].values)
            start = counter
            for value in vals:
                xpos.append(counter)
                colors.append(color_dict2[layer])
                values.append(value)
                counter += 1
            block_centers[layer] = np.mean(range(start, start + len(vals))) if len(vals) > 0 else counter
            counter += 1

        plt.bar(x=xpos, height=values, color=colors, width=0.8, zorder=2)
        for layer in ordered_layers:
            vals = df[df["layer"] == layer]["cosine_similarity"].values
            if len(vals) == 0:
                continue
            mean_val = np.mean(vals)
            center_offset = block_centers[layer] + self._get_offset_cosine_similarity(generic_label, layer)
            plt.axhline(y=mean_val, color=dark_palette[layer], linestyle="--", linewidth=1, zorder=999)
            plt.text(
                center_offset,
                mean_val,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=20,
                color=dark_palette[layer],
                zorder=1000,
            )

        plt.grid(axis="y", linestyle="--", linewidth=0.6, color="lightgray", zorder=0)
        plt.xticks(ticks=[block_centers[layer] for layer in ordered_layers], labels=ordered_layers, fontsize=16)
        plt.xlabel("")
        plt.ylabel("")
        plt.title("")
        plt.tight_layout()
        plt.savefig(
            f"{self.dm.path_cosine_similarity}{self.file_prefix}_{metric}_"
            f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_"
            f"normalized_{str(normalized)}_single_layer_metrics_cosine_similarity_barchart_{generic_label}.png",
            dpi=800,
        )
        plt.show()
