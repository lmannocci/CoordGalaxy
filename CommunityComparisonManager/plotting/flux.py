"""Flux plotting for overlapping-community analysis."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common_variables import (
    co_action_abbreviation_map,
    co_action_column_print,
    color_dict,
    dpi,
    dtype,
    palette,
)
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class OverlappingFluxPlotter:
    """Build and plot community/user flux derived from overlap matrices."""

    def _get_max_y_axis_limit(self, type_aggregation: str, flux_df: pd.DataFrame, generic_label: str) -> float:
        """
        Compute a consistent y-axis maximum for flux plots.

        :param type_aggregation: [str] Aggregation type: communities or users.
        :param flux_df: [pd.DataFrame] Flux dataframe to plot.
        :param generic_label: [str] Generic layer currently plotted.
        :return: [float] Maximum y-axis value with padding.
        """
        if generic_label in co_action_column_print.values():
            flux_df_generic = flux_df[flux_df["generic"] == generic_label]
            if type_aggregation == "communities":
                frequency_df = flux_df_generic.groupby(["layer", "label"]).size().unstack(fill_value=0)
                return max(0, frequency_df.sum(axis=1).max()) + 1

            max_y = flux_df_generic.groupby(["layer", "generic"])[
                ["lost_nodes", "common_nodes", "gained_nodes"]
            ].sum().sum(axis=1).max()
            y_offset = 5000 if self.dataset_name == "uk" else 500
            return max_y + y_offset

        if type_aggregation == "communities":
            max_y = 0
            for generic_value in ["multimodal", "flat_weighted_sum_louvain"]:
                flux_df_generic = flux_df[flux_df["generic"] == generic_value]
                frequency_df = flux_df_generic.groupby(["layer", "label"]).size().unstack(fill_value=0)
                max_y = max(max_y, frequency_df.sum(axis=1).max())
            return max_y + 1

        flux_df_generic = flux_df[flux_df["generic"].isin(["multimodal", "flat_weighted_sum_louvain"])]
        max_y = flux_df_generic.groupby(["layer", "generic"])[
            ["lost_nodes", "common_nodes", "gained_nodes"]
        ].sum().sum(axis=1).max()
        y_offset = 5000 if self.dataset_name == "uk" else 1000
        return max_y + y_offset

    def _plot_flux_df(self, type_aggregation: str, flux_df: pd.DataFrame, mid_th: float, metric: str) -> None:
        """
        Plot stacked flux charts from a saved flux dataframe.

        :param type_aggregation: [str] Aggregation type: communities or users.
        :param flux_df: [pd.DataFrame] Flux dataframe.
        :param mid_th: [float] Threshold used to label matched communities as common.
        :param metric: [str] Overlap metric used for matching.
        :return: None. Figures are saved to disk.
        """
        flux_df["generic"] = flux_df["generic"].replace(co_action_column_print)
        flux_df["layer"] = flux_df["layer"].replace(co_action_column_print)

        for generic_label in flux_df["generic"].unique():
            self.lm.printl(f"{file_name}. plotting flux for generic_label: {generic_label}")
            flux_df_generic = flux_df[flux_df["generic"] == generic_label]
            if generic_label in color_dict.keys():
                flux_df_generic = flux_df_generic[flux_df_generic["layer"] != generic_label]

            if type_aggregation == "communities":
                frequency_df = flux_df_generic.groupby(["layer", "label"]).size().unstack(fill_value=0)
                label_order = ["lost", "common", "gained"]
                filename = (
                    f"{self.dm.path_overlapping_stacked_plot}{self.file_prefix}_{generic_label}_{metric}_"
                    f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}.png"
                )
            else:
                frequency_df = flux_df_generic.groupby("layer")[["lost_nodes", "common_nodes", "gained_nodes"]].sum()
                label_order = ["lost_nodes", "common_nodes", "gained_nodes"]
                filename = (
                    f"{self.dm.path_overlapping_stacked_plot}{self.file_prefix}_{generic_label}_{metric}_"
                    f"th_size_{str(self.community_size_th)}.png"
                )

            reordered_columns = [col for col in label_order if col in frequency_df.columns]
            frequency_df = frequency_df[reordered_columns]
            layer_order = list(co_action_column_print.values())
            if generic_label in co_action_column_print.values():
                layer_order.remove(generic_label)
            frequency_df = frequency_df.loc[layer_order]
            frequency_df.index = pd.CategoricalIndex(frequency_df.index, categories=layer_order, ordered=True)

            ax = frequency_df.plot(
                kind="bar",
                stacked=True,
                figsize=(8, 6),
                color=palette.values(),
                legend=False,
                width=0.9,
            )
            plt.ylim(0, self._get_max_y_axis_limit(type_aggregation, flux_df, generic_label))
            plt.xlabel("")
            plt.ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=16)
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray")

            for container in ax.containers:
                if type_aggregation == "communities":
                    labels = [int(v) if v > 0 else "" for v in container.datavalues]
                    fontsize = 16
                else:
                    th_plot_y = 1000 if self.dataset_name == "uk" else 100
                    labels = [int(v) if v >= th_plot_y else "" for v in container.datavalues]
                    fontsize = 14
                ax.bar_label(container, labels=labels, label_type="center", fontsize=fontsize)

            plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.show()

    def _instantiation_data_flux(self, metric: str, type_aggregation: str) -> dict[str, list[Any]]:
        """
        Initialize the output dictionary used to build flux dataframes.

        :param metric: [str] Overlap metric used for matching.
        :param type_aggregation: [str] Aggregation type: communities or users.
        :return: [dict[str, list[Any]]] Empty flux dictionary.
        """
        if type_aggregation == "users" and metric != "absolute":
            raise ValueError(f"'{type_aggregation}' is compatible only with 'absolute' metric.")
        if type_aggregation == "communities":
            return {"layer": [], "generic": [], "com_layer": [], "com_generic": [], "communities": [], "label": []}
        return {
            "layer": [],
            "generic": [],
            "com_layer": [],
            "com_generic": [],
            "communities": [],
            "lost_nodes": [],
            "common_nodes": [],
            "gained_nodes": [],
        }

    def _create_flux_df_communities(
        self,
        data_flux: dict[str, list[Any]],
        filter_matrix: np.ndarray,
        filter_communities_label_x: np.ndarray,
        filter_communities_label_y: np.ndarray,
        ca_abbr: str,
        abbr_generic_label: str,
        type_ca: str,
        generic_label: str,
        mid_th: float,
        matches: list[tuple[int, int]],
        unmatched_rows: list[int],
        unmatched_cols: list[int],
    ) -> dict[str, list[Any]]:
        """
        Add community-level lost/common/gained records to the flux dictionary.

        :param data_flux: [dict[str, list[Any]]] Output flux dictionary.
        :param filter_matrix: [np.ndarray] Filtered overlap matrix.
        :param filter_communities_label_x: [np.ndarray] Generic community labels.
        :param filter_communities_label_y: [np.ndarray] Layer community labels.
        :param ca_abbr: [str] Abbreviation of the layer co-action.
        :param abbr_generic_label: [str] Abbreviation of the generic layer.
        :param type_ca: [str] Layer co-action name.
        :param generic_label: [str] Generic layer name.
        :param mid_th: [float] Threshold used to label matched communities as common.
        :param matches: [list[tuple[int, int]]] Matched row/column indices.
        :param unmatched_rows: [list[int]] Unmatched layer-community row indices.
        :param unmatched_cols: [list[int]] Unmatched generic-community column indices.
        :return: [dict[str, list[Any]]] Updated flux dictionary.
        """
        for i, j in matches:
            com_x = filter_communities_label_x[j]
            com_y = filter_communities_label_y[i]
            if filter_matrix[i, j] > mid_th:
                data_flux["com_layer"].append(com_y)
                data_flux["com_generic"].append(com_x)
                data_flux["communities"].append(f"{com_x}_{com_y}")
                data_flux["label"].append("common")
            else:
                data_flux["com_layer"].append(com_y)
                data_flux["com_generic"].append(np.nan)
                data_flux["communities"].append(f"{ca_abbr}_{com_y}")
                data_flux["label"].append("lost")

                data_flux["com_layer"].append(np.nan)
                data_flux["com_generic"].append(com_x)
                data_flux["communities"].append(f"{abbr_generic_label}_{com_x}")
                data_flux["label"].append("gained")

            repeat = 1 if filter_matrix[i, j] > mid_th else 2
            data_flux["layer"].extend([type_ca] * repeat)
            data_flux["generic"].extend([generic_label] * repeat)

        for i in unmatched_rows:
            com_y = filter_communities_label_y[i]
            data_flux["com_layer"].append(com_y)
            data_flux["com_generic"].append(np.nan)
            data_flux["communities"].append(f"{ca_abbr}_{com_y}")
            data_flux["label"].append("lost")
            data_flux["layer"].append(type_ca)
            data_flux["generic"].append(generic_label)

        for j in unmatched_cols:
            com_x = filter_communities_label_x[j]
            data_flux["com_layer"].append(np.nan)
            data_flux["com_generic"].append(com_x)
            data_flux["communities"].append(f"{abbr_generic_label}_{com_x}")
            data_flux["label"].append("gained")
            data_flux["layer"].append(type_ca)
            data_flux["generic"].append(generic_label)
        return data_flux

    def _add_set_to_dict(
        self,
        input_set: set,
        label: str,
        com_y: str | float,
        com_x: str | float,
        com: str,
        type_ca: str,
        generic_label: str,
        node_label_dict: dict[str, list[Any]],
    ) -> None:
        """
        Add node labels for lost/common/gained user-level flux.

        :param input_set: [set] User identifiers to append.
        :param label: [str] Flux label: lost, common, or gained.
        :param com_y: [str | float] Layer community label.
        :param com_x: [str | float] Generic community label.
        :param com: [str] Matched-community identifier.
        :param type_ca: [str] Layer co-action name.
        :param generic_label: [str] Generic layer name.
        :param node_label_dict: [dict[str, list[Any]]] Node-level output dictionary.
        :return: None. The dictionary is mutated in place.
        """
        node_label_dict["userId"].extend(input_set)
        node_label_dict["label"].extend([label] * len(input_set))
        node_label_dict["com_layer"].extend([com_y] * len(input_set))
        node_label_dict["com_generic"].extend([com_x] * len(input_set))
        node_label_dict["communities"].extend([com] * len(input_set))
        node_label_dict["layer"].extend([type_ca] * len(input_set))
        node_label_dict["generic"].extend([generic_label] * len(input_set))

    def _create_flux_df_users(
        self,
        data_flux: dict[str, list[Any]],
        filter_matrix: np.ndarray,
        filter_community_size_x: np.ndarray,
        filter_community_size_y: np.ndarray,
        filter_communities_label_x: np.ndarray,
        filter_communities_label_y: np.ndarray,
        ca_abbr: str,
        abbr_generic_label: str,
        type_ca: str,
        generic_label: str,
        matches: list[tuple[int, int]],
        unmatched_rows: list[int],
        unmatched_cols: list[int],
        filter_intersection_matrix: np.ndarray,
        filter_x_set: np.ndarray,
        filter_y_set: np.ndarray,
        node_label_dict: dict[str, list[Any]],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """
        Add user-level lost/common/gained records to flux dictionaries.

        :param data_flux: [dict[str, list[Any]]] Output flux dictionary.
        :param filter_matrix: [np.ndarray] Filtered overlap matrix.
        :param filter_community_size_x: [np.ndarray] Generic community sizes.
        :param filter_community_size_y: [np.ndarray] Layer community sizes.
        :param filter_communities_label_x: [np.ndarray] Generic community labels.
        :param filter_communities_label_y: [np.ndarray] Layer community labels.
        :param ca_abbr: [str] Abbreviation of the layer co-action.
        :param abbr_generic_label: [str] Abbreviation of the generic layer.
        :param type_ca: [str] Layer co-action name.
        :param generic_label: [str] Generic layer name.
        :param matches: [list[tuple[int, int]]] Matched row/column indices.
        :param unmatched_rows: [list[int]] Unmatched layer-community row indices.
        :param unmatched_cols: [list[int]] Unmatched generic-community column indices.
        :param filter_intersection_matrix: [np.ndarray] Filtered intersection sets.
        :param filter_x_set: [np.ndarray] Filtered generic community user sets.
        :param filter_y_set: [np.ndarray] Filtered layer community user sets.
        :param node_label_dict: [dict[str, list[Any]]] Node-level output dictionary.
        :return: [tuple] Updated flux and node-label dictionaries.
        """
        for i, j in matches:
            com_x = filter_communities_label_x[j]
            com_y = filter_communities_label_y[i]
            data_flux["com_layer"].append(com_y)
            data_flux["com_generic"].append(com_x)
            data_flux["communities"].append(f"{com_x}_{com_y}")
            data_flux["layer"].append(type_ca)
            data_flux["generic"].append(generic_label)
            data_flux["lost_nodes"].append(filter_community_size_y[i] - filter_matrix[i, j])
            data_flux["common_nodes"].append(filter_matrix[i, j])
            data_flux["gained_nodes"].append(filter_community_size_x[j] - filter_matrix[i, j])

            self._add_set_to_dict(
                filter_y_set[i] - filter_intersection_matrix[i, j],
                "lost",
                com_y,
                np.nan,
                f"{com_x}_{com_y}",
                type_ca,
                generic_label,
                node_label_dict,
            )
            self._add_set_to_dict(
                filter_intersection_matrix[i, j],
                "common",
                com_y,
                com_x,
                f"{com_x}_{com_y}",
                type_ca,
                generic_label,
                node_label_dict,
            )
            self._add_set_to_dict(
                filter_x_set[j] - filter_intersection_matrix[i, j],
                "gained",
                np.nan,
                com_x,
                f"{com_x}_{com_y}",
                type_ca,
                generic_label,
                node_label_dict,
            )

        for i in unmatched_rows:
            com_y = filter_communities_label_y[i]
            data_flux["com_layer"].append(com_y)
            data_flux["com_generic"].append(np.nan)
            data_flux["communities"].append(f"{ca_abbr}_{com_y}")
            data_flux["layer"].append(type_ca)
            data_flux["generic"].append(generic_label)
            data_flux["lost_nodes"].append(filter_community_size_y[i])
            data_flux["common_nodes"].append(0)
            data_flux["gained_nodes"].append(0)
            self._add_set_to_dict(
                filter_y_set[i],
                "lost",
                com_y,
                np.nan,
                f"{ca_abbr}_{com_y}",
                type_ca,
                generic_label,
                node_label_dict,
            )

        for j in unmatched_cols:
            com_x = filter_communities_label_x[j]
            data_flux["com_layer"].append(np.nan)
            data_flux["com_generic"].append(com_x)
            data_flux["communities"].append(f"{abbr_generic_label}_{com_x}")
            data_flux["layer"].append(type_ca)
            data_flux["generic"].append(generic_label)
            data_flux["lost_nodes"].append(0)
            data_flux["common_nodes"].append(0)
            data_flux["gained_nodes"].append(filter_community_size_x[j])
            self._add_set_to_dict(
                filter_x_set[j],
                "gained",
                np.nan,
                com_x,
                f"{abbr_generic_label}_{com_x}",
                type_ca,
                generic_label,
                node_label_dict,
            )

        return data_flux, node_label_dict

    def _create_flux_df(
        self,
        data_flux: dict[str, list[Any]],
        filter_matrix: np.ndarray,
        filter_community_size_x: np.ndarray,
        filter_community_size_y: np.ndarray,
        filter_communities_label_x: np.ndarray,
        filter_communities_label_y: np.ndarray,
        ca_abbr: str,
        abbr_generic_label: str,
        type_ca: str,
        generic_label: str,
        mid_th: float,
        type_aggregation: str,
        matches: list[tuple[int, int]],
        unmatched_rows: list[int],
        unmatched_cols: list[int],
        filter_intersection_matrix: np.ndarray,
        filter_x_set: np.ndarray,
        filter_y_set: np.ndarray,
        node_label_dict: dict[str, list[Any]],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]] | None]:
        """
        Dispatch flux-dataframe construction by aggregation type.

        :param data_flux: [dict[str, list[Any]]] Output flux dictionary.
        :param filter_matrix: [np.ndarray] Filtered overlap matrix.
        :param filter_community_size_x: [np.ndarray] Generic community sizes.
        :param filter_community_size_y: [np.ndarray] Layer community sizes.
        :param filter_communities_label_x: [np.ndarray] Generic community labels.
        :param filter_communities_label_y: [np.ndarray] Layer community labels.
        :param ca_abbr: [str] Abbreviation of the layer co-action.
        :param abbr_generic_label: [str] Abbreviation of the generic layer.
        :param type_ca: [str] Layer co-action name.
        :param generic_label: [str] Generic layer name.
        :param mid_th: [float] Threshold used to label matched communities as common.
        :param type_aggregation: [str] Aggregation type: communities or users.
        :param matches: [list[tuple[int, int]]] Matched row/column indices.
        :param unmatched_rows: [list[int]] Unmatched row indices.
        :param unmatched_cols: [list[int]] Unmatched column indices.
        :param filter_intersection_matrix: [np.ndarray] Filtered intersection sets.
        :param filter_x_set: [np.ndarray] Filtered generic community user sets.
        :param filter_y_set: [np.ndarray] Filtered layer community user sets.
        :param node_label_dict: [dict[str, list[Any]]] Node-level output dictionary.
        :return: [tuple] Updated flux and node-label dictionaries.
        """
        if type_aggregation == "communities":
            data_flux = self._create_flux_df_communities(
                data_flux,
                filter_matrix,
                filter_communities_label_x,
                filter_communities_label_y,
                ca_abbr,
                abbr_generic_label,
                type_ca,
                generic_label,
                mid_th,
                matches,
                unmatched_rows,
                unmatched_cols,
            )
            return data_flux, None

        return self._create_flux_df_users(
            data_flux,
            filter_matrix,
            filter_community_size_x,
            filter_community_size_y,
            filter_communities_label_x,
            filter_communities_label_y,
            ca_abbr,
            abbr_generic_label,
            type_ca,
            generic_label,
            matches,
            unmatched_rows,
            unmatched_cols,
            filter_intersection_matrix,
            filter_x_set,
            filter_y_set,
            node_label_dict,
        )

    @log_method
    def plot_stacked_flux(
        self,
        type_aggregation: str,
        mid_th: float = 0.5,
        metric: str = "harmonicMean",
        plot_heatmap_list: list[str] | None = None,
    ) -> None:
        """
        Build and plot gained/common/lost community or user flux from saved overlap tensors.

        :param type_aggregation: [str] Aggregation type: communities or users.
        :param mid_th: [float] Threshold used to label matched communities as common.
        :param metric: [str] Overlap metric used for matching.
        :param plot_heatmap_list: [list[str] | None] Optional generic labels to plot.
        :return: None. Flux CSVs and plots are saved in the overlapping-analysis directories.
        """
        overlapping_tensor = self.ch.load_object(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p"
        )
        overlapping_set_tensor = self.ch.load_object(
            self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_set.p"
        )

        try:
            data_flux = self._instantiation_data_flux(metric, type_aggregation)
        except ValueError as error:
            self.lm.printl(error)
            return

        node_label_dict = {
            "userId": [],
            "label": [],
            "layer": [],
            "generic": [],
            "com_layer": [],
            "com_generic": [],
            "communities": [],
        }
        for (x_label, y_label), overlapping_matrix_collection in overlapping_tensor.items():
            if x_label in self.available_list_ca and y_label not in self.available_list_ca:
                type_ca, generic_label = x_label, y_label
            elif x_label not in self.available_list_ca and y_label in self.available_list_ca:
                type_ca, generic_label = y_label, x_label
            elif x_label in self.available_list_ca and y_label in self.available_list_ca:
                type_ca, generic_label = y_label, x_label
            else:
                continue

            if plot_heatmap_list is not None and generic_label not in plot_heatmap_list:
                continue

            self.lm.printl(f"{file_name}. plot_stacked_flux {type_aggregation} ({x_label}, {y_label}).")
            overlapping_set = overlapping_set_tensor[(x_label, y_label)]
            ca_abbr = co_action_abbreviation_map[type_ca]
            abbr_generic_label = co_action_abbreviation_map[generic_label] if generic_label in self.available_list_ca else generic_label

            matrix_object = overlapping_matrix_collection[metric]
            matrix = matrix_object["matrix"]
            communities_label_x = np.array(matrix_object["x_label"])
            communities_label_y = np.array(matrix_object["y_label"])
            community_size_x = np.array(matrix_object["x_size_community"])
            community_size_y = np.array(matrix_object["y_size_community"])
            intersection_matrix = np.array(overlapping_set["overlapping_set"])
            x_set = np.array(overlapping_set["x_set"])
            y_set = np.array(overlapping_set["y_set"])

            (
                filter_matrix,
                filter_community_size_x,
                filter_community_size_y,
                filter_communities_label_x,
                filter_communities_label_y,
                filter_intersection_matrix,
                filter_x_set,
                filter_y_set,
            ) = self.matrix_filter.filter_communities(
                matrix,
                community_size_x,
                community_size_y,
                communities_label_x,
                communities_label_y,
                self.community_size_th,
                intersection_matrix,
                x_set,
                y_set,
            )

            matches, unmatched_rows, unmatched_cols, _total_similarity = self.overlap_metric_calculator.max_similarity_match(
                filter_matrix
            )
            data_flux, node_label_dict = self._create_flux_df(
                data_flux,
                filter_matrix,
                filter_community_size_x,
                filter_community_size_y,
                filter_communities_label_x,
                filter_communities_label_y,
                ca_abbr,
                abbr_generic_label,
                type_ca,
                generic_label,
                mid_th,
                type_aggregation,
                matches,
                unmatched_rows,
                unmatched_cols,
                filter_intersection_matrix,
                filter_x_set,
                filter_y_set,
                node_label_dict,
            )

        data_flux_df = pd.DataFrame(data_flux)
        node_label_df = pd.DataFrame(node_label_dict)
        if type_aggregation == "communities":
            filename = (
                f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv"
            )
        else:
            filename = (
                f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_"
                f"th_size_{str(self.community_size_th)}_flux_df.csv"
            )
            self.ch.save_dataframe(
                node_label_df,
                f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_th_size_{str(self.community_size_th)}_node_labelling.csv",
            )

        self.ch.save_dataframe(data_flux_df, filename)
        data_flux_df = self.ch.read_dataframe(filename, dtype=dtype)
        self._plot_flux_df(type_aggregation, data_flux_df, mid_th, metric)
