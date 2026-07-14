import ast
from typing import Any

import pandas as pd


class MentionHashtagPreprocessor:
    """Shared helpers for mention and hashtag list extraction."""

    def __init__(self, logger: Any, file_name: str) -> None:
        """
        Store logger context used by extraction helpers.

        :param logger: LogManager instance used for progress logging.
        :param file_name: Name of the caller module used in logs.
        :return: None.
        """
        self.lm = logger
        self.file_name = file_name

    def extract_element(self, original_list: list[dict], column: str) -> list:
        """
        Extract a named field from every dictionary in a list.

        :param original_list: List of dictionaries to process.
        :param column: Dictionary key to extract from each element.
        :return: List of extracted values.
        """
        return [element[column] for element in original_list]

    def ensure_list(self, value: Any) -> list:
        """
        Return a real list from a list value or a stringified list.

        :param value: Value that may already be a list or a string representation of a list.
        :return: Parsed list, or an empty list when parsing is not possible.
        """
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
                if isinstance(parsed_value, list):
                    return parsed_value
            except (ValueError, SyntaxError):
                return []
        return []

    def has_content_list(self, value: Any) -> bool:
        """
        Return whether a value contains at least one mention or hashtag item.

        :param value: Value to test for list-like content.
        :return: True when the value contains at least one item, otherwise False.
        """
        return (
            isinstance(value, list) and len(value) > 0
        ) or (
            isinstance(value, str) and value.strip() not in ['[]', '']
        )

    def build_list_object_dataframe(self, filter_df: pd.DataFrame, list_column: str) -> pd.DataFrame:
        """
        Explode a list column and rename the extracted values to objectId.

        :param filter_df: Dataframe containing a list column to explode.
        :param list_column: Name of the list column to explode.
        :return: Dataframe with one objectId per row.
        """
        self.lm.printl(f"{self.file_name}. explode {list_column}.")
        result_df = filter_df.explode(list_column)
        result_df = result_df.rename(columns={list_column: "objectId"})
        result_df = result_df[result_df['objectId'].notna() & (result_df['objectId'] != '')]
        result_df['objectId'] = result_df['objectId'].astype(str).str.strip().str.lower()
        return result_df.dropna()
