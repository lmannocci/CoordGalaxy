from __future__ import annotations

from typing import Any

import pandas as pd

from utils.common_variables import action_map_inverse, custom_flatten_algorithm


class CommunityDataPreparer:
    """
    Prepare community dataframes for cross-community comparisons.
    """

    def get_labels_reordered(
        self,
        type_algorithm_x: str,
        type_algorithm_y: str,
        list_ca_x: list[Any],
        list_ca_y: list[Any],
        cda_x: Any,
        cda_y: Any,
        df_x: pd.DataFrame | None = None,
        df_y: pd.DataFrame | None = None,
    ) -> tuple[Any, ...]:
        """
        Return comparison labels and reorder dataframes so multimodal output is on the x axis.

        :param type_algorithm_x: [str] Algorithm type for x, one-layer or multi-layer.
        :param type_algorithm_y: [str] Algorithm type for y, one-layer or multi-layer.
        :param list_ca_x: [list[Any]] Co-actions for x.
        :param list_ca_y: [list[Any]] Co-actions for y.
        :param cda_x: [Any] Community-detection algorithm for x.
        :param cda_y: [Any] Community-detection algorithm for y.
        :param df_x: [pd.DataFrame | None] Optional x community dataframe.
        :param df_y: [pd.DataFrame | None] Optional y community dataframe.
        :return: Labels and, when provided, reordered dataframes.
        """
        x_label = self._community_output_label(type_algorithm_x, list_ca_x, cda_x)
        y_label = self._community_output_label(type_algorithm_y, list_ca_y, cda_y)

        if df_x is None or df_y is None:
            if x_label == "multimodal":
                return x_label, y_label
            if y_label == "multimodal":
                return y_label, x_label
            return x_label, y_label

        if x_label == "multimodal":
            return x_label, y_label, self._normalize_multimodal_df(df_x), df_y
        if y_label == "multimodal":
            return y_label, x_label, self._normalize_multimodal_df(df_y), df_x
        return x_label, y_label, df_x, df_y

    def filter_community_size(self, df: pd.DataFrame, community_size_th: int) -> pd.DataFrame:
        """
        Keep only communities with at least the requested number of rows.

        :param df: [pd.DataFrame] Community dataframe with a group column.
        :param community_size_th: [int] Minimum community size.
        :return: [pd.DataFrame] Filtered dataframe.
        """
        agg_df = df.groupby(["group"]).size().reset_index(name="nUsers")
        left_com_df = pd.DataFrame(agg_df[agg_df["nUsers"] >= community_size_th]["group"])
        return pd.merge(df, left_com_df, on="group", how="inner")

    def extract_user_sets(
        self,
        x_label: str,
        y_label: str,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract one user-set dataframe per compared community output.

        :param x_label: [str] Label for x community output.
        :param y_label: [str] Label for y community output.
        :param df_x: [pd.DataFrame] X community dataframe.
        :param df_y: [pd.DataFrame] Y community dataframe.
        :return: [tuple[pd.DataFrame, pd.DataFrame]] Dataframes with group and userSet columns.
        """
        user_set_df_x = self._extract_side_user_sets(df_x, x_label, y_label)
        user_set_df_y = self._extract_side_user_sets(df_y, y_label, x_label)

        user_set_df_x = user_set_df_x.sort_values(by="group").rename(columns={"userId": "userSet"})
        user_set_df_y = user_set_df_y.sort_values(by="group").rename(columns={"userId": "userSet"})
        return user_set_df_x, user_set_df_y

    def _community_output_label(self, type_algorithm: str, list_ca: list[Any], cda: Any) -> str:
        """
        Return the label used to compare one community output.

        :param type_algorithm: [str] Algorithm type, one-layer or multi-layer.
        :param list_ca: [list[Any]] Co-actions in the output.
        :param cda: [Any] Community-detection algorithm.
        :return: [str] Comparison label.
        """
        if type_algorithm == "one-layer":
            return list_ca[0].get_co_action()
        if cda.get_algorithm_name() in custom_flatten_algorithm:
            return cda.get_algorithm_name()
        return "multimodal"

    def _normalize_multimodal_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a multiplex community dataframe to group/userId/layer columns.

        :param df: [pd.DataFrame] Multiplex community dataframe.
        :return: [pd.DataFrame] Normalized dataframe.
        """
        df = df.rename(columns={"cid": "group", "actor": "userId"}).copy()
        df["layer"] = df["layer"].map(action_map_inverse)
        return df

    def _extract_side_user_sets(self, df: pd.DataFrame, own_label: str, other_label: str) -> pd.DataFrame:
        """
        Extract user sets for one side of a comparison.

        :param df: [pd.DataFrame] Community dataframe.
        :param own_label: [str] Label for this side.
        :param other_label: [str] Label for the compared side.
        :return: [pd.DataFrame] Dataframe with group and userId set columns.
        """
        group_list = df["group"].unique()
        if own_label == "multimodal":
            user_set_df = pd.DataFrame(df.groupby(["group", "layer"])["userId"].apply(set)).reset_index()
            user_set_df = user_set_df[user_set_df["layer"] == other_label][["group", "userId"]]
            excluded_communities = list(set(group_list) - set(user_set_df["group"].unique()))
            new_rows = pd.DataFrame({
                "group": excluded_communities,
                "userId": [set() for _ in range(len(excluded_communities))],
            })
            return pd.concat([user_set_df, new_rows], ignore_index=True)
        return df.groupby("group")["userId"].apply(set).reset_index()
