"""Validation analysis for overlapping-community comparisons."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.common_variables import dpi, dtype, flatten_algorithm, multimodal_print, one_layer_algorithm, palette
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class ValidationAnalysisMixin:
    """Combine, compute, and plot validation labels for overlap analysis."""

    def _add_coordination_label(
        self,
        df: pd.DataFrame,
        col: str = "percCoord",
        thre_co: float = 0.68,
        thre_not: float = 0.10,
    ) -> pd.DataFrame:
        """
        Add a categorical coordination label from a percentage-coordination column.

        :param df: [pd.DataFrame] Validation dataframe.
        :param col: [str] Column containing coordination percentage.
        :param thre_co: [float] Threshold above which a community is coordinated.
        :param thre_not: [float] Threshold below which a community is not coordinated.
        :return: [pd.DataFrame] Dataframe with labelCoordination column.
        """
        def classify(value: float) -> str:
            """
            Convert one coordination percentage into a validation label.

            :param value: [float] Coordination percentage.
            :return: [str] Validation label.
            """
            if value >= thre_co:
                return "coordinated"
            if value <= thre_not:
                return "notCoordinated"
            return "mixed"

        df = df.copy()
        df["labelCoordination"] = df[col].apply(classify)
        return df

    def _validation_by_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count validation categories per generic/layer/overlap-label tuple.

        :param df: [pd.DataFrame] Flux dataframe enriched with layer and generic validation labels.
        :return: [pd.DataFrame] Counts of validation labels by overlap label.
        """
        results = []
        for (generic, layer, label), group in df.groupby(["generic", "layer", "label"]):
            row = {"generic": generic, "layer": layer, "label": label}
            if label == "common" or label == "gained":
                value_counts = dict(group["labelCoordination_generic"].value_counts())
            elif label == "lost":
                value_counts = dict(group["labelCoordination_layer"].value_counts())
            else:
                value_counts = {}
            row.update(value_counts)
            results.append(row)

        return pd.DataFrame(results).fillna(0)

    @log_method
    def combine_validation_communities(self, cda) -> None:
        """
        Combine validation metrics produced by single-layer or multiplex community detection.

        :param cda: Community-detection algorithm object used to locate analysis outputs.
        :return: None. Combined validation rows are saved under the overlap validation path.
        """
        if cda.get_algorithm_name() in one_layer_algorithm:
            df_list = []
            th_str = "" if self.community_size_th is None else f"th_size_{str(self.community_size_th)}_"
            for type_ca, dict_path in self.dm.dict_path_ca.items():
                df = self.ch.read_dataframe(
                    f"{dict_path['path_filter_community']}{repr(cda)}{os.sep}analysis{os.sep}"
                    f"{type_ca}_{th_str}validation_communities.csv",
                    dtype=dtype,
                )
                df["layer"] = type_ca
                df_list.append(df)
            combined_df = pd.concat(df_list, ignore_index=True)
        elif cda.get_algorithm_name() in flatten_algorithm:
            combined_df = self.ch.read_dataframe(
                f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}"
                f"{cda.get_algorithm_name()}_validation_communities.csv",
                dtype=dtype,
            )
            combined_df["layer"] = cda.get_algorithm_name()
        else:
            combined_df = self.ch.read_dataframe(
                f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}"
                f"{cda.get_algorithm_name()}_group_isControl_validation_communities.csv",
                dtype=dtype,
            )
            combined_df["layer"] = cda.get_algorithm_name()

        self.ch.update_dataframe(
            combined_df,
            f"{self.dm.path_validation}{self.file_prefix}_validation_communities.csv",
            dtype=dtype,
        )

    @log_method
    def compute_validation_by_label(self, mid_th: float = 0.5, metric: str = "harmonicMean") -> None:
        """
        Compute validation-category counts for lost/common/gained overlap labels.

        :param mid_th: [float] Flux threshold encoded in the input filename.
        :param metric: [str] Overlap metric encoded in the input filename.
        :return: None. The validation-label summary is saved to disk.
        """
        flux_df = self._read_overlap_flux_df(metric, mid_th)
        validation_df = self.ch.read_dataframe(
            self.dm.path_validation + f"{self.file_prefix}_validation_communities.csv",
            dtype=dtype,
        )
        validation_df = self._add_coordination_label(validation_df)
        filter_df = validation_df[["group", "layer", "labelCoordination"]].copy()

        filter_df["layer"].replace({"glouvain": "multimodal", "ginfomap": "multimodal"}, inplace=True)
        filter_df.rename(columns={"group": "com_layer"}, inplace=True)
        merged_df = pd.merge(flux_df, filter_df, on=["layer", "com_layer"], how="left")
        merged_df = merged_df.rename(columns={"labelCoordination": "labelCoordination_layer"})

        filter_df.rename(columns={"com_layer": "com_generic", "layer": "generic"}, inplace=True)
        merged_df = pd.merge(merged_df, filter_df, on=["generic", "com_generic"], how="left")
        merged_df = merged_df.rename(columns={"labelCoordination": "labelCoordination_generic"})

        result_df = self._validation_by_label(merged_df)
        self.ch.save_dataframe(result_df, f"{self.dm.path_validation}{self.file_prefix}_validation_label.csv")

    @log_method
    def plot_validation_multimodal(self, cda) -> None:
        """
        Plot validation summaries for multimodal and flattened community assignments.

        :param cda: Community-detection algorithm object used to select flattened rows.
        :return: None. Validation plots are saved to disk.
        """
        validation_label_df = self.ch.read_dataframe(
            f"{self.dm.path_validation}{self.file_prefix}_validation_label.csv",
            dtype=dtype,
        )
        aggr_generic = (
            validation_label_df.groupby(["generic", "label"])[["notCoordinated", "coordinated"]]
            .sum()
            .reset_index()
        )
        aggr_generic["percCoord"] = aggr_generic["coordinated"] / (
            aggr_generic["coordinated"] + aggr_generic["notCoordinated"]
        )
        mul_df = aggr_generic[
            (aggr_generic["generic"] == "multimodal") | (aggr_generic["generic"] == cda.get_algorithm_name())
        ].copy()
        mul_df["generic"] = mul_df["generic"].replace(multimodal_print)

        value2plot_dict = {
            "notCoordinated": "not-coordinated count",
            "coordinated": "coordinated count",
            "percCoord": "percentage coordinated",
        }
        for value2plot in value2plot_dict:
            self._plot_validation_value(cda, mul_df, value2plot)

    def _plot_validation_value(self, cda, mul_df: pd.DataFrame, value2plot: str) -> None:
        """
        Plot one validation bar chart.

        :param cda: Community-detection algorithm object used in the output filename.
        :param mul_df: [pd.DataFrame] Aggregated validation dataframe.
        :param value2plot: [str] Validation column to plot.
        :return: None. The figure is saved to disk.
        """
        label_order = ["lost", "common", "gained"]
        df_plot = mul_df.copy()
        df_plot["label"] = pd.Categorical(df_plot["label"], categories=label_order, ordered=True)
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            data=df_plot,
            x="generic",
            y=value2plot,
            hue="label",
            palette=palette,
            order=sorted(df_plot["generic"].unique(), reverse=True),
            legend=False,
            width=0.9,
        )
        self._annotate_validation_bars(ax, value2plot)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
        plt.xlabel("", fontsize=16)
        plt.ylabel("", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            f"{self.dm.path_validation}{self.file_prefix}_multimodal_{cda.get_algorithm_name()}_"
            f"validation_{value2plot}.png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()

    def _annotate_validation_bars(self, ax, value2plot: str) -> None:
        """
        Add readable labels to validation bar containers.

        :param ax: Matplotlib axis containing the barplot.
        :param value2plot: [str] Validation column being plotted.
        :return: None. The axis is updated in place.
        """
        for container in ax.containers:
            if self.file_prefix == "louvain_resolution_1":
                th_plot_y = 0.04 if value2plot == "percCoord" else 4
            elif self.file_prefix == "infomap":
                th_plot_y = 0.05 if value2plot == "percCoord" else 0
            else:
                th_plot_y = 0

            if value2plot == "percCoord":
                labels = [round(value, 2) if value > th_plot_y else "" for value in container.datavalues]
            else:
                labels = [int(value) if value > th_plot_y else "" for value in container.datavalues]
            ax.bar_label(container, labels=labels, padding=3, fontsize=16, label_type="center")
