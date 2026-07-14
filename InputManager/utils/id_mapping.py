import json
import os
from typing import Any

import pandas as pd


class UserIdMapper:
    """Persist and apply a stable mapping from original user ids to framework ids."""

    def __init__(self, mapping_path: str) -> None:
        """
        Create a mapper backed by a JSON mapping file.

        :param mapping_path: Path to the JSON file storing id mappings.
        :return: None.
        """
        self.mapping_path = mapping_path

    def load_mapping(self) -> dict[str, str]:
        """
        Load the current id mapping, or return an empty mapping when absent.

        :return: Dictionary mapping original ids to framework ids.
        """
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as file:
                return json.load(file)

        return {}

    def load_existing_mapping(self) -> dict[str, str]:
        """
        Load an existing id mapping and fail when the mapping file is missing.

        :return: Dictionary mapping original ids to framework ids.
        """
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")

        return self.load_mapping()

    def load_reverse_mapping(self) -> dict[str, str]:
        """
        Load the reverse mapping from framework ids to original ids.

        :return: Dictionary mapping framework ids to original ids.
        """
        mapping = self.load_existing_mapping()
        return {mapped_value: original_value for original_value, mapped_value in mapping.items()}

    def save_mapping(self, mapping: dict[str, str]) -> None:
        """
        Persist the id mapping to disk.

        :param mapping: Dictionary mapping original ids to framework ids.
        :return: None.
        """
        with open(self.mapping_path, "w", encoding="utf-8") as file:
            json.dump(mapping, file, indent=2, ensure_ascii=False)

    def map_value(self, value: Any, mapping: dict[str, str], mapped_values: set[str] | None = None) -> str:
        """
        Map one source value to an existing or newly assigned user id.

        :param value: Source id value to map.
        :param mapping: Mutable dictionary of existing id mappings.
        :param mapped_values: Optional set of already assigned framework ids for faster batch mapping.
        :return: Framework id assigned to the source value.
        """
        value = str(value)

        if mapped_values is None:
            mapped_values = set(mapping.values())

        if value in mapped_values:
            return value

        if value not in mapping:
            mapping[value] = f"u_{len(mapping)}"
            mapped_values.add(mapping[value])

        return mapping[value]

    def map_user_ids(self, df: pd.DataFrame, column: str = "userId") -> pd.DataFrame:
        """
        Map a dataframe user id column when the column is present.

        :param df: Dataframe containing a user id column.
        :param column: Name of the column to map.
        :return: Dataframe with mapped ids, or the original dataframe when the column is absent.
        """
        if column not in df.columns:
            return df

        mapping = self.load_mapping()
        mapped_values = set(mapping.values())
        df = df.copy()
        df[column] = df[column].astype(str).apply(lambda value: self.map_value(value, mapping, mapped_values))
        self.save_mapping(mapping)
        return df

    def map_series(self, series: pd.Series) -> pd.Series:
        """
        Map a series of user-like ids using the same persisted mapping.

        :param series: Series of ids to map.
        :return: Series of mapped framework ids.
        """
        mapping = self.load_mapping()
        mapped_values = set(mapping.values())
        mapped_series = series.astype(str).apply(lambda value: self.map_value(value, mapping, mapped_values))
        self.save_mapping(mapping)
        return mapped_series

    def original_to_simple_user_id(self, user_id: Any) -> str | None:
        """
        Convert one original user id to its simple mapped id.

        :param user_id: Original user id value.
        :return: Simple mapped id, or None when the id is not in the mapping.
        """
        mapping = self.load_existing_mapping()
        return mapping.get(str(user_id))

    def simple_to_original_user_id(self, simple_user_id: Any) -> str | None:
        """
        Convert one simple mapped id back to its original user id.

        :param simple_user_id: Simple mapped user id.
        :return: Original user id, or None when the id is not in the mapping.
        """
        reverse_mapping = self.load_reverse_mapping()
        return reverse_mapping.get(str(simple_user_id))

    def restore_original_user_ids(self, df: pd.DataFrame, column: str = "userId") -> pd.DataFrame:
        """
        Restore original user ids in a dataframe column.

        :param df: Dataframe containing simple mapped ids.
        :param column: Name of the column to restore.
        :return: Copy of the dataframe with original ids restored.
        """
        reverse_mapping = self.load_reverse_mapping()
        df = df.copy()
        df[column] = df[column].map(reverse_mapping)
        return df

    def convert_user_ids_to_simple_ids(self, df: pd.DataFrame, column: str = "userId") -> pd.DataFrame:
        """
        Convert original user ids to simple mapped ids in a dataframe column.

        :param df: Dataframe containing original ids.
        :param column: Name of the column to convert.
        :return: Copy of the dataframe with simple mapped ids.
        """
        mapping = self.load_existing_mapping()
        df = df.copy()
        df[column] = df[column].astype(str).map(mapping)
        return df

    def convert_edge_list(self, edge_list: list[tuple], direction: str = "to_simple") -> list[tuple]:
        """
        Convert the first two user ids in every edge-list tuple.

        :param edge_list: Edge list tuples where the first two elements are user ids.
        :param direction: Conversion direction: "to_simple" or "restore".
        :return: Edge list with converted user ids.
        """
        if direction == "to_simple":
            mapping = self.load_existing_mapping()
            mapping = {
                **mapping,
                **{key.replace("-", "_"): value for key, value in mapping.items()}
            }
        elif direction == "restore":
            mapping = self.load_reverse_mapping()
        else:
            raise ValueError("direction must be either 'to_simple' or 'restore'")

        converted_edge_list = []
        for user_id1, user_id2, *rest in edge_list:
            user_id1_str = str(user_id1)
            user_id2_str = str(user_id2)
            converted_edge_list.append((
                mapping.get(user_id1_str, user_id1_str),
                mapping.get(user_id2_str, user_id2_str),
                *rest
            ))

        return converted_edge_list
