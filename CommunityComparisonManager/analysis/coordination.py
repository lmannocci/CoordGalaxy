"""Coordination analysis for overlapping-community comparisons."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.common_variables import dpi, dtype, one_layer_algorithm, palette
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class CoordinationAnalysisMixin:
    """Compute and plot coordination statistics for overlap labels."""

    def _compute_weight_stats_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute average coordination-weight statistics per generic layer and flux label.

        :param df: [pd.DataFrame] Flux rows merged with layer and generic coordination metrics.
        :return: [pd.DataFrame] Summary with mean average, median, std, and MAD weights.
        """
        metrics = ["avg_weight", "median_weight", "std_weight", "mad_weight"]
        results = []

        for (generic, label), group in df.groupby(["generic", "label"]):
            row = {"generic": generic, "label": label}
            for metric in metrics:
                col_layer = f"{metric}_layer"
                col_generic = f"{metric}_generic"
                if label == "common":
                    values = pd.concat([group[col_layer], group[col_generic]])
                elif label == "gained":
                    values = group[col_generic]
                elif label == "lost":
                    values = group[col_layer]
                else:
                    values = pd.Series(dtype=float)
                row[f"avg_{metric}"] = values.mean() if len(values) > 0 else np.nan
            results.append(row)

        return pd.DataFrame(results)

    def _get_shift_coordination_mean(self, generic: str, label: str) -> float:
        """
        Return custom x-axis shift for coordination mean labels.

        :param generic: [str] Generic layer label.
        :param label: [str] Flux label.
        :return: [float] Horizontal annotation shift.
        """
        shift = 0
        if self.dataset_name == "uk" and self.file_prefix == "infomap":
            if generic == "flat_weighted_sum_infomap" and label == "common":
                shift = -2.5
            elif generic == "multimodal" and label == "gained":
                shift = -1
        elif self.dataset_name == "uk" and self.file_prefix == "louvain_resolution_1":
            if generic == "flat_weighted_sum_louvain" and (label == "lost" or label == "common"):
                shift = -3
            elif generic == "multimodal" and label == "gained":
                shift = -4
        elif self.dataset_name == "russia1" and self.file_prefix == "infomap":
            if generic == "flat_weighted_sum_infomap" and (label == "common" or label == "gained"):
                shift = -2.5
        elif self.dataset_name == "russia1" and self.file_prefix == "louvain_resolution_1":
            if generic == "flat_weighted_sum_louvain" and label == "lost":
                shift = -2
        return shift

    def _merge_flux_with_coordination(self, flux_df: pd.DataFrame, coord_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge flux labels with coordination metrics for layer and generic communities.

        :param flux_df: [pd.DataFrame] Community-level flux dataframe.
        :param coord_df: [pd.DataFrame] Coordination metrics dataframe.
        :return: [pd.DataFrame] Flux dataframe enriched with layer and generic coordination columns.
        """
        coord_df = coord_df.copy()
        coord_df["layer"].replace({"glouvain": "multimodal", "ginfomap": "multimodal"}, inplace=True)
        coord_df.rename(columns={"community": "com_layer"}, inplace=True)
        merged_df = pd.merge(flux_df, coord_df, on=["layer", "com_layer"], how="left")
        merged_df = merged_df.rename(
            columns={
                "size": "size_layer",
                "avg_weight": "avg_weight_layer",
                "std_weight": "std_weight_layer",
                "median_weight": "median_weight_layer",
                "mad_weight": "mad_weight_layer",
            }
        )

        coord_df.rename(columns={"com_layer": "com_generic", "layer": "generic"}, inplace=True)
        merged_df = pd.merge(merged_df, coord_df, on=["generic", "com_generic"], how="left")
        return merged_df.rename(
            columns={
                "size": "size_generic",
                "avg_weight": "avg_weight_generic",
                "std_weight": "std_weight_generic",
                "median_weight": "median_weight_generic",
                "mad_weight": "mad_weight_generic",
            }
        )

    @log_method
    def combine_coordination_communities(self, cda) -> None:
        """
        Combine coordination metrics produced by single-layer or multiplex community detection.

        :param cda: Community-detection algorithm object used to locate analysis outputs.
        :return: None. Combined coordination rows are saved under the overlap validation path.
        """
        if cda.get_algorithm_name() in one_layer_algorithm:
            df_list = []
            th_str = "" if self.community_size_th is None else f"th_size_{str(self.community_size_th)}_"
            for type_ca, dict_path in self.dm.dict_path_ca.items():
                df = self.ch.read_dataframe(
                    f"{dict_path['path_filter_community']}{repr(cda)}{os.sep}analysis{os.sep}"
                    f"{type_ca}_{th_str}coordination_communities.csv",
                    dtype=dtype,
                )
                df["layer"] = type_ca
                df_list.append(df)
            combined_df = pd.concat(df_list, ignore_index=True)
        elif cda.get_algorithm_name() in {"flat_nw_infomap", "flat_nw_louvain"}:
            self.lm.printl(f"{file_name}. combine_coordination_communities skipped, algorithm: {cda.get_algorithm_name()}.")
            return
        else:
            combined_df = self.ch.read_dataframe(
                f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}"
                f"{cda.get_algorithm_name()}_coordination_communities.csv",
                dtype=dtype,
            )
            combined_df["layer"] = cda.get_algorithm_name()

        self.ch.update_dataframe(
            combined_df,
            f"{self.dm.path_validation}{self.file_prefix}_coordination_communities.csv",
            dtype=dtype,
        )

    @log_method
    def compute_coordination_by_label(self, mid_th: float = 0.5, metric: str = "harmonicMean") -> None:
        """
        Compute coordination-weight summaries for lost/common/gained overlap labels.

        :param mid_th: [float] Flux threshold encoded in the input filename.
        :param metric: [str] Overlap metric encoded in the input filename.
        :return: None. The coordination-label summary is saved to disk.
        """
        flux_df = self._read_overlap_flux_df(metric, mid_th)
        coord_df = self.ch.read_dataframe(
            self.dm.path_validation + f"{self.file_prefix}_coordination_communities.csv",
            dtype=dtype,
        )
        merged_df = self._merge_flux_with_coordination(flux_df, coord_df)
        coordination_label_df = self._compute_weight_stats_summary(merged_df)
        self.ch.save_dataframe(coordination_label_df, f"{self.dm.path_validation}{self.file_prefix}_coordination_label.csv")

    @log_method
    def plot_coordination_by_label(
        self,
        cda,
        metric: str = "harmonicMean",
        mid_th: float = 0.5,
        metric_to_plot: str = "avg_weight",
    ) -> None:
        """
        Plot coordination metric distributions by overlap label.

        :param cda: Community-detection algorithm object used to select multiplex/flattened rows.
        :param metric: [str] Overlap metric encoded in the flux filename.
        :param mid_th: [float] Flux threshold encoded in the flux filename.
        :param metric_to_plot: [str] Coordination metric prefix to plot.
        :return: None. Coordination plots are saved to disk.
        """
        flux_df = self._read_overlap_flux_df(metric, mid_th)
        coord_df = self.ch.read_dataframe(
            self.dm.path_validation + f"{self.file_prefix}_coordination_communities.csv",
            dtype=dtype,
        )
        merged_df = self._merge_flux_with_coordination(flux_df, coord_df)
        mul_df = merged_df[
            (merged_df["generic"] == "multimodal") | (merged_df["generic"] == cda.get_algorithm_name())
        ].copy()
        label_order = ["lost", "common", "gained"]
        dark_palette = {label: tuple(channel * 0.7 for channel in color) for label, color in palette.items()}

        rows = []
        for (generic, label), group in mul_df.groupby(["generic", "label"]):
            if label == "common":
                values = list(group[f"{metric_to_plot}_layer"]) + list(group[f"{metric_to_plot}_generic"])
            elif label == "gained":
                values = list(group[f"{metric_to_plot}_generic"])
            elif label == "lost":
                values = list(group[f"{metric_to_plot}_layer"])
            else:
                values = []
            for value in sorted(values):
                rows.append({"generic": generic, "label": label, "value": value})

        plot_df = pd.DataFrame(rows)
        plot_df["label"] = pd.Categorical(plot_df["label"], categories=label_order, ordered=True)
        for generic, gdf in plot_df.groupby("generic"):
            self._plot_coordination_group(generic, gdf, label_order, dark_palette, metric_to_plot)

    def _plot_coordination_group(
        self,
        generic: str,
        gdf: pd.DataFrame,
        label_order: list[str],
        dark_palette: dict[str, tuple[float, ...]],
        metric_to_plot: str,
    ) -> None:
        """
        Plot one coordination bar chart for a generic layer.

        :param generic: [str] Generic layer label.
        :param gdf: [pd.DataFrame] Plot dataframe for that generic layer.
        :param label_order: [list[str]] Ordered overlap labels.
        :param dark_palette: [dict[str, tuple[float, ...]]] Dark colors used for mean annotations.
        :param metric_to_plot: [str] Coordination metric prefix being plotted.
        :return: None. The figure is saved to disk.
        """
        plt.figure(figsize=(12, 5), dpi=300)
        xpos = []
        colors = []
        values = []
        counter = 0
        block_centers = {}

        for label in label_order:
            vals = gdf[gdf["label"] == label]["value"].values
            start = counter
            for value in vals:
                xpos.append(counter)
                colors.append(palette[label])
                values.append(value)
                counter += 1
            block_centers[label] = np.mean(range(start, start + len(vals))) if len(vals) > 0 else counter
            counter += 1

        plt.bar(x=xpos, height=values, color=colors, width=0.8, zorder=2)
        for label in label_order:
            vals = gdf[gdf["label"] == label]["value"].values
            if len(vals) == 0:
                continue
            mean_val = np.mean(vals)
            plt.axhline(y=mean_val, color=dark_palette[label], linestyle="--", linewidth=1, zorder=999)
            center_label_shifted = block_centers[label] + self._get_shift_coordination_mean(generic, label)
            plt.text(
                center_label_shifted,
                mean_val,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=20,
                color=dark_palette[label],
                zorder=1000,
            )

        plt.grid(axis="y", linestyle="--", linewidth=0.5, color="lightgray", zorder=0)
        plt.xticks(ticks=[block_centers[label] for label in label_order], labels=label_order, fontsize=16)
        plt.xlabel("", fontsize=16)
        plt.ylabel("", fontsize=16)
        plt.tight_layout()
        filename = f"{self.dm.path_validation}{self.file_prefix}_{generic}_{metric_to_plot}_coordination.png"
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
        self.lm.printl(f"{file_name}. {filename} saved.")
        plt.show()
