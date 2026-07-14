"""Overlap tensor computation for community-comparison analyses."""

from __future__ import annotations

import os

import numpy as np

from utils.common_variables import available_overlapping_metrics, dtype
from utils.decorator_definition import log_method


file_name = os.path.splitext(os.path.basename(__file__))[0]


class OverlapComputationMixin:
    """Compute and save overlap matrices between two community assignments."""

    @log_method
    def compute_overlapping(self, save_overlapping_tensor: bool = True, save_intersections: bool = False) -> None:
        """
        Compute overlap matrices between two community assignments.

        :param save_overlapping_tensor: [bool] Whether to save overlap metric matrices.
        :param save_intersections: [bool] Whether to save intersection sets.
        :return: None. Outputs are saved to the overlapping-analysis directory.
        """
        df_x = self.ch_x.read_dataframe(self.dm_x.path_user_dataframe + "com_df.csv", dtype=dtype)
        df_y = self.ch_y.read_dataframe(self.dm_y.path_user_dataframe + "com_df.csv", dtype=dtype)
        x_label, y_label, df_x, df_y = self.community_data_preparer.get_labels_reordered(
            self.type_algorithm_x,
            self.type_algorithm_y,
            self.list_ca_x,
            self.list_ca_y,
            self.cda_x,
            self.cda_y,
            df_x,
            df_y,
        )
        self.lm.printl(f"{file_name}. unique layer df_x: {df_x['layer'].unique()}.")
        if self.community_size_th is not None:
            df_x = self.community_data_preparer.filter_community_size(df_x, self.community_size_th)
            df_y = self.community_data_preparer.filter_community_size(df_y, self.community_size_th)
            self.file_prefix = f"{self.file_prefix}_th_{str(self.community_size_th)}"

        self.lm.printl(f"{file_name}. computing overlap for ({x_label}, {y_label}).")
        overlapping_tensor = self._load_existing_tensor(f"{self.file_prefix}_overlapping_tensor.p")
        overlapping_set_tensor = self._load_existing_tensor(f"{self.file_prefix}_overlapping_set.p")

        n_com_x = len(df_x["group"].unique())
        n_com_y = len(df_y["group"].unique())
        total_combination = n_com_x * n_com_y
        self.lm.printl(f"{file_name}. communities: {n_com_x} x {n_com_y} = {total_combination} combinations.")

        user_set_df_x, user_set_df_y = self.community_data_preparer.extract_user_sets(x_label, y_label, df_x, df_y)
        overlapping_matrix_numpy = self._initialize_overlap_matrices(user_set_df_x, user_set_df_y, df_x, df_y)
        overlapping_set = self._initialize_overlap_sets(user_set_df_x, user_set_df_y)

        count = 0
        for _index_y, row_y in user_set_df_y.iterrows():
            c_y = row_y["userSet"]
            overlapping_set["y_set"].append(c_y)
            row_dict = {metric: [] for metric in available_overlapping_metrics}
            row_intersections = []
            for _index_x, row_x in user_set_df_x.iterrows():
                c_x = row_x["userSet"]
                overlapping_dict, intersection = self.overlap_metric_calculator.compute_overlap(c_x, c_y)
                overlapping_set["x_set"].append(c_x)
                row_intersections.append(intersection)
                for metric, value in overlapping_dict.items():
                    row_dict[metric].append(value)
                count += 1
                self.lm.printK(count, 100, f"{file_name}. overlap {count}/{total_combination} completed.")

            for metric, row in row_dict.items():
                overlapping_matrix_numpy[metric]["matrix"].append(row)
            overlapping_set["overlapping_set"].append(row_intersections)

        for metric in overlapping_matrix_numpy.keys():
            overlapping_matrix_numpy[metric]["matrix"] = np.array(overlapping_matrix_numpy[metric]["matrix"])

        overlapping_tensor[(x_label, y_label)] = overlapping_matrix_numpy
        overlapping_set_tensor[(x_label, y_label)] = overlapping_set

        if save_overlapping_tensor:
            self.ch.save_object(
                overlapping_tensor,
                self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_tensor.p",
            )
        if save_intersections:
            self.ch.save_object(
                overlapping_set_tensor,
                self.dm.path_overlapping_analysis + f"{self.file_prefix}_overlapping_set.p",
            )

    def _load_existing_tensor(self, filename: str) -> dict:
        """
        Load an existing overlap tensor if present.

        :param filename: [str] Tensor filename inside the overlapping-analysis directory.
        :return: [dict] Loaded tensor or an empty dictionary.
        """
        path = self.dm.path_overlapping_analysis + filename
        if os.path.exists(path):
            return self.ch.load_object(path)
        return {}

    def _initialize_overlap_matrices(self, user_set_df_x, user_set_df_y, df_x, df_y) -> dict:
        """
        Initialize overlap matrix containers for every overlap metric.

        :param user_set_df_x: [pd.DataFrame] User sets for the x-side communities.
        :param user_set_df_y: [pd.DataFrame] User sets for the y-side communities.
        :param df_x: [pd.DataFrame] X-side community dataframe.
        :param df_y: [pd.DataFrame] Y-side community dataframe.
        :return: [dict] Matrix containers keyed by overlap metric.
        """
        return {
            metric: {
                "matrix": [],
                "x_label": list(user_set_df_x["group"].unique()),
                "y_label": list(user_set_df_y["group"].unique()),
                "x_size_community": df_x.groupby("group").size().values,
                "y_size_community": df_y.groupby("group").size().values,
            }
            for metric in available_overlapping_metrics
        }

    def _initialize_overlap_sets(self, user_set_df_x, user_set_df_y) -> dict:
        """
        Initialize containers for overlap intersection sets.

        :param user_set_df_x: [pd.DataFrame] User sets for the x-side communities.
        :param user_set_df_y: [pd.DataFrame] User sets for the y-side communities.
        :return: [dict] Set containers used by user-flux analyses.
        """
        return {
            "x_set": [],
            "y_set": [],
            "overlapping_set": [],
            "x_label": list(user_set_df_x["group"].unique()),
            "y_label": list(user_set_df_y["group"].unique()),
        }
