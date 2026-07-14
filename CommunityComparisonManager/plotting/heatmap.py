"""Heatmap plotting for overlapping-community analysis."""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.common_variables import co_action_column_print, dpi, dtype, multimodal_print
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class OverlappingHeatmapPlotter:
    """Plot heatmaps generated from overlapping-community matrices."""

    def _concat_single_layer_matrix(
        self,
        overlapping_tensor: dict[tuple[str, str], dict[str, dict[str, Any]]],
        concat_matrix_bool: bool,
        plot_heatmap_list: list[str] | None,
    ) -> tuple[dict, dict, dict, dict, dict, dict]:
        """
        Prepare concatenated or separated overlap matrices for heatmap plotting.

        :param overlapping_tensor: [dict] Saved overlap matrices indexed by compared labels.
        :param concat_matrix_bool: [bool] Whether single-layer matrices should be vertically concatenated.
        :param plot_heatmap_list: [list[str] | None] Optional generic labels to keep.
        :return: [tuple] Concatenated matrices, metadata, and separated matrices.
        """
        concat_matrix = defaultdict(lambda: defaultdict(dict))
        separate_matrix = {}
        concat_matrix_n_rows = defaultdict(lambda: defaultdict(list))
        single_layer_co_actions = defaultdict(lambda: defaultdict(list))
        community_labels = defaultdict(lambda: defaultdict(lambda: {"x_label": [], "y_label": []}))
        community_sizes = defaultdict(lambda: defaultdict(lambda: {"x_size_community": [], "y_size_community": []}))

        for (x_label, y_label), overlapping_matrix_collection in overlapping_tensor.items():
            if x_label in self.available_list_ca and y_label in self.available_list_ca:
                type_ca, generic_label = y_label, x_label
            elif x_label in self.available_list_ca:
                type_ca, generic_label = x_label, y_label
            elif y_label in self.available_list_ca:
                type_ca, generic_label = y_label, x_label
            else:
                continue

            if plot_heatmap_list is not None and generic_label not in plot_heatmap_list:
                continue

            if concat_matrix_bool:
                for metric, matrix_object in overlapping_matrix_collection.items():
                    matrix = matrix_object["matrix"]
                    if metric not in concat_matrix[generic_label]:
                        concat_matrix[generic_label][metric] = matrix
                    else:
                        self.lm.printl(
                            f"{x_label}, {y_label}, {metric}, "
                            f"{str(concat_matrix[generic_label][metric].shape)}, {matrix.shape}"
                        )
                        concat_matrix[generic_label][metric] = np.concatenate(
                            [concat_matrix[generic_label][metric], matrix],
                            axis=0,
                        )

                    concat_matrix_n_rows[generic_label][metric].append(matrix.shape[0])
                    single_layer_co_actions[generic_label][metric].append(type_ca)
                    community_labels[generic_label][metric]["y_label"].extend(matrix_object["y_label"])
                    community_labels[generic_label][metric]["x_label"] = matrix_object["x_label"]
                    community_sizes[generic_label][metric]["x_size_community"] = matrix_object["x_size_community"]
                    community_sizes[generic_label][metric]["y_size_community"].extend(
                        matrix_object["y_size_community"]
                    )
            else:
                separate_matrix[(x_label, y_label)] = overlapping_matrix_collection

        return (
            concat_matrix,
            concat_matrix_n_rows,
            single_layer_co_actions,
            community_labels,
            community_sizes,
            separate_matrix,
        )

    def _create_plot_heatmap(
        self,
        concat_heatmap: bool,
        filter_matrix: np.ndarray,
        filter_communities_label_x: np.ndarray,
        filter_communities_label_y: np.ndarray,
        x_label: str,
        metric: str,
        scale_factor_height: float = 1,
        n_matrix: int | None = None,
        filter_height_matrix_list: np.ndarray | None = None,
        filter_y_label_list: np.ndarray | None = None,
        y_label: str | None = None,
    ) -> None:
        """
        Render and save one overlap heatmap.

        :param concat_heatmap: [bool] Whether the heatmap contains concatenated single-layer matrices.
        :param filter_matrix: [np.ndarray] Matrix after community-size filtering.
        :param filter_communities_label_x: [np.ndarray] X-axis community labels.
        :param filter_communities_label_y: [np.ndarray] Y-axis community labels.
        :param x_label: [str] Generic comparison label for the x-axis.
        :param metric: [str] Overlap metric name.
        :param scale_factor_height: [float] Height multiplier for stacked matrices.
        :param n_matrix: [int | None] Number of concatenated matrices.
        :param filter_height_matrix_list: [np.ndarray | None] Heights of concatenated matrices.
        :param filter_y_label_list: [np.ndarray | None] Layer labels for concatenated matrices.
        :param y_label: [str | None] Single-layer y-axis label for separated heatmaps.
        :return: None. The heatmap is saved to disk.
        """
        plt.figure(figsize=(len(filter_communities_label_x), len(filter_communities_label_y) * scale_factor_height))
        reversed_cmap = sns.color_palette("Blues", as_cmap=True)
        ax = sns.heatmap(
            filter_matrix,
            cmap=reversed_cmap,
            cbar=True,
            xticklabels=filter_communities_label_x,
            yticklabels=filter_communities_label_y,
            linewidths=0.5,
            linecolor="white",
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        max_value = np.max(filter_matrix)
        min_value = np.min(filter_matrix)
        mid_value = (max_value - min_value) / 2

        for i in range(filter_matrix.shape[0]):
            for j in range(filter_matrix.shape[1]):
                if filter_matrix[i, j] > 0.035:
                    color = "white" if filter_matrix[i, j] > mid_value else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        self.matrix_filter.annotate_format(filter_matrix[i, j]),
                        color=color,
                        ha="center",
                        va="center",
                        fontsize=12,
                    )

        x_label_print = co_action_column_print[x_label] if x_label in self.available_list_ca else multimodal_print[x_label]
        ax.set_xlabel(x_label_print)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=16)

        if concat_heatmap:
            cumulative_height = 0
            for i, height in enumerate(filter_height_matrix_list):
                cumulative_height += height
                if i < n_matrix - 1:
                    plt.hlines(cumulative_height, *plt.xlim(), colors="black", linewidth=2)

            cumulative_height = 0
            for i, current_y_label in enumerate(filter_y_label_list):
                y_label_print = co_action_column_print[current_y_label]
                matrix_height = filter_height_matrix_list[i]
                midpoint = cumulative_height + matrix_height / 2
                plt.text(-0.5, midpoint, y_label_print, va="center", ha="right", fontsize=16, rotation=90)
                cumulative_height += matrix_height

            filename = (
                f"{self.dm.path_overlapping_heatmap}{self.file_prefix}_{x_label}_single_layer_"
                f"{metric}_th_size_{str(self.community_size_th)}.png"
            )
        else:
            ax.set_ylabel(co_action_column_print[y_label])
            filename = (
                f"{self.dm.path_overlapping_heatmap}{self.file_prefix}_{x_label}_{y_label}_"
                f"{metric}_th_size_{str(self.community_size_th)}.png"
            )

        plt.gca().set_aspect("auto")
        plt.tight_layout(pad=0)
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.show()

    def _plot_concat_matrix(
        self,
        concat_matrix: dict,
        concat_matrix_n_rows: dict,
        single_layer_co_actions: dict,
        community_labels: dict,
        community_sizes: dict,
        scale_factor_height: float,
    ) -> None:
        """
        Plot concatenated single-layer overlap heatmaps.

        :param concat_matrix: [dict] Matrices grouped by generic label and metric.
        :param concat_matrix_n_rows: [dict] Matrix heights grouped by generic label and metric.
        :param single_layer_co_actions: [dict] Layer labels for each concatenated matrix block.
        :param community_labels: [dict] Community labels grouped by generic label and metric.
        :param community_sizes: [dict] Community sizes grouped by generic label and metric.
        :param scale_factor_height: [float] Height multiplier for plotting.
        :return: None. Heatmaps are saved to disk.
        """
        for x_label, overlapping_matrix_collection in concat_matrix.items():
            for metric, matrix in overlapping_matrix_collection.items():
                self.lm.printl(
                    f"{file_name}. {x_label} vs single layer, metric: {metric}, "
                    f"filtered community_size_th >= {str(self.community_size_th)}."
                )
                height_matrix_list = np.array(concat_matrix_n_rows[x_label][metric])
                n_matrix = len(height_matrix_list)
                y_label_list = np.array(single_layer_co_actions[x_label][metric])
                communities_label_x = np.array(community_labels[x_label][metric]["x_label"])
                communities_label_y = np.array(community_labels[x_label][metric]["y_label"])
                community_size_x = np.array(community_sizes[x_label][metric]["x_size_community"])
                community_size_y = np.array(community_sizes[x_label][metric]["y_size_community"])

                (
                    filter_matrix,
                    _filter_community_size_x,
                    _filter_community_size_y,
                    filter_communities_label_x,
                    filter_communities_label_y,
                    filter_height_matrix_list,
                    filter_y_label_list,
                ) = self.matrix_filter.filter_communities_and_height_matrix_list(
                    matrix,
                    community_size_x,
                    community_size_y,
                    communities_label_x,
                    communities_label_y,
                    height_matrix_list,
                    y_label_list,
                    self.community_size_th,
                )

                if filter_matrix.size != 0:
                    self._create_plot_heatmap(
                        True,
                        filter_matrix,
                        filter_communities_label_x,
                        filter_communities_label_y,
                        x_label,
                        metric,
                        scale_factor_height=scale_factor_height,
                        n_matrix=n_matrix,
                        filter_height_matrix_list=filter_height_matrix_list,
                        filter_y_label_list=filter_y_label_list,
                    )
                else:
                    self.lm.printl(
                        f"{file_name}. {x_label} vs single layer, "
                        f"filtered community_size_th >= {str(self.community_size_th)} gives an empty matrix."
                    )

    def _plot_separate_matrix(self, separate_matrix: dict, scale_factor_height: float) -> None:
        """
        Plot separated pairwise overlap heatmaps.

        :param separate_matrix: [dict] Matrices grouped by compared labels.
        :param scale_factor_height: [float] Height multiplier for plotting.
        :return: None. Heatmaps are saved to disk.
        """
        for (x_label, y_label), overlapping_matrix_collection in separate_matrix.items():
            for metric, matrix_object in overlapping_matrix_collection.items():
                community_size_x = np.array(matrix_object["x_size_community"])
                community_size_y = np.array(matrix_object["y_size_community"])
                matrix = matrix_object["matrix"]
                communities_label_x = np.array(matrix_object["x_label"])
                communities_label_y = np.array(matrix_object["y_label"])

                (
                    filter_matrix,
                    _filter_community_size_x,
                    _filter_community_size_y,
                    filter_communities_label_x,
                    filter_communities_label_y,
                ) = self.matrix_filter.filter_communities(
                    matrix,
                    community_size_x,
                    community_size_y,
                    communities_label_x,
                    communities_label_y,
                    self.community_size_th,
                )

                self._create_plot_heatmap(
                    False,
                    filter_matrix,
                    filter_communities_label_x,
                    filter_communities_label_y,
                    x_label,
                    metric,
                    scale_factor_height=scale_factor_height,
                    y_label=y_label,
                )

    @log_method
    def plot_heatmap_overlapping_matrix(
        self,
        concat_matrix_bool: bool = True,
        plot_heatmap_list: list[str] | None = None,
    ) -> None:
        """
        Plot heatmaps from saved overlap matrices.

        :param concat_matrix_bool: [bool] Whether to concatenate single-layer matrices into one heatmap.
        :param plot_heatmap_list: [list[str] | None] Optional generic labels to plot.
        :return: None. Heatmaps are saved in the overlapping heatmap directory.
        """
        overlapping_tensor = self.ch.load_object(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p"
        )
        (
            concat_matrix,
            concat_matrix_n_rows,
            single_layer_co_actions,
            community_labels,
            community_sizes,
            separate_matrix,
        ) = self._concat_single_layer_matrix(overlapping_tensor, concat_matrix_bool, plot_heatmap_list)
        self._plot_concat_matrix(
            concat_matrix,
            concat_matrix_n_rows,
            single_layer_co_actions,
            community_labels,
            community_sizes,
            0.7,
        )
        self._plot_separate_matrix(separate_matrix, 0.7)

    @log_method
    def plot_heatmap_single_layer_NMI(self) -> None:
        """
        Plot a heatmap of single-layer NMI scores.

        :return: None. Heatmap is saved in the overlapping NMI directory.
        """
        if self.community_size_th is not None:
            prefix = f"{self.file_prefix}_th_{str(self.community_size_th)}"
        else:
            prefix = self.file_prefix

        df = self.ch.read_dataframe(
            self.dm.path_overlapping_NMI + f"{prefix}_single_layer_NMI.csv",
            dtype=dtype,
        )
        df["layer_x"] = df["layer_x"].replace(co_action_column_print)
        df["layer_y"] = df["layer_y"].replace(co_action_column_print)

        layer_order = list(co_action_column_print.values())
        df["layer_x"] = pd.Categorical(df["layer_x"], categories=layer_order, ordered=True)
        df["layer_y"] = pd.Categorical(df["layer_y"], categories=layer_order, ordered=True)
        heatmap_data = df.pivot(index="layer_x", columns="layer_y", values="NMI_score")
        mask = np.triu(np.ones_like(heatmap_data, dtype=bool), k=1)
        reversed_cmap = sns.color_palette("viridis", as_cmap=True).reversed()

        plt.figure(figsize=(8, 6.5))
        ax = sns.heatmap(
            heatmap_data,
            mask=mask,
            annot=True,
            cmap=reversed_cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 16},
            cbar_kws={"label": "NMI score"},
            vmin=0,
            vmax=0.4,
        )

        for i in range(len(heatmap_data)):
            diagonal_value = heatmap_data.iloc[i, i]
            if pd.notna(diagonal_value):
                diagonal_value = int(diagonal_value)
                ax.add_patch(
                    plt.Rectangle(
                        (i + 0.01, i + 0.01),
                        0.978,
                        0.978,
                        fill=True,
                        color="black",
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                ax.text(
                    i + 0.5,
                    i + 0.5,
                    f"{diagonal_value}",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=16,
                )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.figure.axes[-1].yaxis.label.set_size(16)
        ax.figure.axes[-1].tick_params(labelsize=12)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

        plt.savefig(
            self.dm.path_overlapping_NMI + f"{prefix}_single_layer_NMI_heatmap.png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()
