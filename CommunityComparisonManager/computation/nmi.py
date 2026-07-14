"""NMI computation for single-layer community comparisons."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from utils.common_variables import dtype
from utils.decorator_definition import log_method


class SingleLayerNMIAnalysisMixin:
    """Compute normalized mutual information between two community assignments."""

    @log_method
    def compute_single_layer_NMI(self) -> None:
        """
        Compute normalized mutual information between two single-layer community assignments.

        :return: None. NMI rows are appended to the overlapping NMI CSV.
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
        if x_label not in self.available_list_ca and y_label not in self.available_list_ca:
            self.lm.printl("Single-layer NMI requires at least one single-layer partition.")
            return

        if self.community_size_th is not None:
            df_x = self.community_data_preparer.filter_community_size(df_x, self.community_size_th)
            df_y = self.community_data_preparer.filter_community_size(df_y, self.community_size_th)
            self.file_prefix = f"{self.file_prefix}_th_{str(self.community_size_th)}"

        if len(df_x) == 0:
            self.lm.printl(
                f"Single-layer NMI skipped: {x_label} is empty after community_size_th={self.community_size_th}."
            )
            return
        if len(df_y) == 0:
            self.lm.printl(
                f"Single-layer NMI skipped: {y_label} is empty after community_size_th={self.community_size_th}."
            )
            return

        common_nodes = set(df_x["userId"]).intersection(set(df_y["userId"]))
        exclusive_x = set(df_x["userId"]) - common_nodes
        exclusive_y = set(df_y["userId"]) - common_nodes
        max_label = max(df_x["group"].max(), df_y["group"].max())

        singleton_x = pd.DataFrame(
            {"userId": list(exclusive_x), "group": range(max_label + 1, max_label + 1 + len(exclusive_x))}
        )
        df_y = pd.concat([df_y, singleton_x], ignore_index=True)
        max_label += len(exclusive_x)

        singleton_y = pd.DataFrame(
            {"userId": list(exclusive_y), "group": range(max_label + 1, max_label + 1 + len(exclusive_y))}
        )
        df_x = pd.concat([df_x, singleton_y], ignore_index=True)

        labels_x = df_x.set_index("userId")["group"].to_dict()
        labels_y = df_y.set_index("userId")["group"].to_dict()
        all_nodes = list(set(labels_x.keys()).union(labels_y.keys()))
        community_labels_x = [labels_x.get(node, -1) for node in all_nodes]
        community_labels_y = [labels_y.get(node, -1) for node in all_nodes]
        nmi_score = normalized_mutual_info_score(community_labels_x, community_labels_y)

        results_df = pd.DataFrame(
            {
                "layer_x": [x_label],
                "layer_y": [y_label],
                "NMI_score": [nmi_score],
            }
        )
        self.ch.update_dataframe(
            results_df,
            self.dm.path_overlapping_NMI + f"{self.file_prefix}_single_layer_NMI.csv",
            dtype=dtype,
        )
