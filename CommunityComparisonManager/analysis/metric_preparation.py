"""Prepare community and node metric files for overlap visualizations."""

from __future__ import annotations

import os

import pandas as pd

from utils.common_variables import dtype, flatten_algorithm, one_layer_algorithm
from utils.decorator_definition import log_method


class OverlapMetricPreparationMixin:
    """Combine metric files produced by community-detection modules."""

    @log_method
    def combine_single_layer_metrics_communities(self, cda) -> None:
        """
        Combine single-layer community structural metrics into one overlap-analysis file.

        :param cda: Community-detection algorithm object used to locate single-layer analysis outputs.
        :return: None. Combined metrics are saved in the overlapping-analysis directory.
        """
        df_list = []
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            df = self.ch.read_dataframe(
                f"{dict_path['path_filter_community']}{repr(cda)}{os.sep}analysis{os.sep}"
                f"{type_ca}_th_size_{str(self.community_size_th)}_metrics_communities.csv",
                dtype=dtype,
            )
            df["layer"] = type_ca
            df = df.rename(columns={"community": "com_layer"})
            df_list.append(df)

        combined_df = pd.concat(df_list, ignore_index=True)
        self.ch.save_dataframe(
            combined_df,
            f"{self.dm.path_overlapping_analysis}{self.file_prefix}_single_layer_metrics_communities.csv",
        )

    @log_method
    def combine_node_metrics(self, cda) -> None:
        """
        Combine node metrics from single-layer or flattened community-detection outputs.

        :param cda: Community-detection algorithm object used to locate analysis outputs.
        :return: None. Combined node metrics are saved in the overlapping-analysis directory.
        """
        if cda.get_algorithm_name() in one_layer_algorithm:
            combined_df = self._combine_one_layer_node_metrics(cda)
        elif cda.get_algorithm_name() in flatten_algorithm:
            combined_df = self._read_flattened_node_metrics(cda)
        else:
            self.lm.printl(f"combine_node_metrics skipped: unsupported algorithm {cda.get_algorithm_name()}.")
            return

        self.ch.update_dataframe(
            combined_df,
            f"{self.dm.path_overlapping_analysis}{self.file_prefix}_node_metrics.csv",
            dtype=dtype,
        )

    def _combine_one_layer_node_metrics(self, cda) -> pd.DataFrame:
        """
        Combine node metrics produced independently for each single-layer co-action.

        :param cda: Community-detection algorithm object used to locate single-layer analysis outputs.
        :return: [pd.DataFrame] Combined node metrics.
        """
        df_list = []
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            df = self.ch.read_dataframe(
                f"{dict_path['path_filter_community']}{repr(cda)}{os.sep}analysis{os.sep}{type_ca}_node_metrics.csv",
                dtype=dtype,
            )
            df = df.rename(columns={"community": "com_layer"})
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    def _read_flattened_node_metrics(self, cda) -> pd.DataFrame:
        """
        Read node metrics produced by a flattened community-detection algorithm.

        :param cda: Community-detection algorithm object used to locate flattened analysis outputs.
        :return: [pd.DataFrame] Node metrics with community renamed to com_layer.
        """
        df = self.ch.read_dataframe(
            f"{self.dm.path_community}{repr(cda)}{os.sep}analysis{os.sep}{cda.get_algorithm_name()}_node_metrics.csv",
            dtype=dtype,
        )
        return df.rename(columns={"community": "com_layer"})
