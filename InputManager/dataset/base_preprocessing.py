from abc import ABC, abstractmethod
from typing import Any, Sequence

import pandas as pd

from ..utils.id_mapping import UserIdMapper
from ..utils.mentions_hashtags_processing import MentionHashtagPreprocessor
from ..utils.urls_processing import URLPreprocessor
from utils.decorator_definition import log_method


class DatasetPreprocessing(ABC):
    """Base contract for transforming any dataset into the framework input schema."""

    def __init__(self, checkpoint: Any, directory_manager: Any, logger: Any, file_name: str) -> None:
        """
        Store shared services and initialize reusable preprocessing utilities.

        :param checkpoint: Checkpoint instance used to read and save artifacts.
        :param directory_manager: DirectoryManager instance with dataset paths.
        :param logger: LogManager instance used for progress logging.
        :param file_name: Name of the caller module used in logs.
        :return: None.
        """
        self.ch = checkpoint
        self.dm = directory_manager
        self.lm = logger
        self.file_name = file_name
        self.url_preprocessor = URLPreprocessor(logger, file_name)
        self.content_preprocessor = MentionHashtagPreprocessor(logger, file_name)
        self.user_id_mapper = UserIdMapper(self.dm.path_temp_data + "user_id_mapping.json")

    @abstractmethod
    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize the main dataset file into a common post-level dataframe.

        :param df: Raw dataset dataframe to normalize.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized dataframe using the common framework columns.
        """
        pass

    def normalize_user_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
        """
        Optionally normalize a dataset-specific users file.

        :param df: Raw users dataframe to normalize.
        :param filename: Name of the normalized users CSV file to save under temp_data.
        :return: Normalized users dataframe, or None when the dataset has no users file.
        """
        self.lm.printl(f"{self.file_name}. normalize_user_data not implemented for this dataset.")
        return None

    def normalize_data_text(self, df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
        """
        Optionally normalize a dataset-specific text file.

        :param df: Raw text dataframe to normalize.
        :param filename: Name of the normalized text CSV file to save under temp_data.
        :return: Normalized text dataframe, or None when the dataset has no separate text file.
        """
        self.lm.printl(f"{self.file_name}. normalize_data_text not implemented for this dataset.")
        return None

    @abstractmethod
    def extract_url_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare raw URL lists from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing id,userId,created,url_list,contentType and optional labels.
        """
        pass

    @abstractmethod
    def extract_hashtag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare raw hashtag lists from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing id,userId,created,hashtag_list,contentType and optional labels.
        """
        pass

    @abstractmethod
    def extract_mention_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare raw mention lists from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing id,userId,created,mention_list,contentType and optional labels.
        """
        pass

    @abstractmethod
    def extract_retweet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare retweet objects from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing retweet objects in an objectId-compatible column.
        """
        pass

    @abstractmethod
    def extract_reply_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare reply objects from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing reply objects in an objectId-compatible column.
        """
        pass

    def extract_comment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally select and prepare comment objects from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing comment objects in objectId, or an empty standard dataframe.
        """
        return self.empty_standard_dataframe(include_is_control="isControl" in df.columns)

    @abstractmethod
    def extract_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare text objects from normalized dataset rows.

        :param df: Normalized dataset dataframe.
        :return: Dataframe containing text objects in an objectId-compatible column.
        """
        pass

    def map_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map the userId column to stable framework-level anonymous ids.

        :param df: Dataframe containing a userId column.
        :return: Copy of the dataframe with mapped userId values.
        """
        return self.user_id_mapper.map_user_ids(df)

    def standardize_normalized_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply shared normalized-data conventions such as contentType and user mapping.

        :param df: Dataset-normalized dataframe that may still contain legacy column names.
        :return: Dataframe with contentType naming and mapped userId values.
        """
        df = df.rename(columns={"type": "contentType"})
        return self.map_user_ids(df)

    def empty_standard_dataframe(self, include_is_control: bool = False) -> pd.DataFrame:
        """
        Return an empty dataframe with the common co-action schema.

        :param include_is_control: Whether to include the optional isControl label column.
        :return: Empty dataframe with standard co-action columns.
        """
        columns = ["id", "userId", "created", "objectId", "contentType"]
        if include_is_control:
            columns.append("isControl")
        return pd.DataFrame(columns=columns)

    def standardize_object_data(
        self,
        df: pd.DataFrame,
        object_column: str | None = None,
        content_type: str | None = None
    ) -> pd.DataFrame:
        """
        Convert extracted co-action data to id,userId,created,objectId,contentType.

        :param df: Extracted co-action dataframe to standardize.
        :param object_column: Optional source column to rename to objectId.
        :param content_type: Optional contentType value to assign to every row.
        :return: Standardized co-action dataframe.
        """
        df = df.copy()

        required_columns = {"id", "userId", "created", "objectId", "contentType"}
        renamed_object_column = object_column and object_column in df.columns
        if df.empty and not (required_columns.issubset(df.columns) or renamed_object_column):
            return self.empty_standard_dataframe(include_is_control="isControl" in df.columns)

        if object_column and object_column in df.columns and object_column != "objectId":
            df = df.rename(columns={object_column: "objectId"})

        if "type" in df.columns and "contentType" not in df.columns:
            df = df.rename(columns={"type": "contentType"})

        if content_type is not None:
            df["contentType"] = content_type

        df = self.map_user_ids(df)

        columns = ["id", "userId", "created", "objectId", "contentType"]
        optional_columns = [column for column in ["isControl"] if column in df.columns]
        return df[columns + optional_columns]

    def save_standard_object_data(
        self,
        df: pd.DataFrame,
        filename: str,
        object_column: str | None = None,
        content_type: str | None = None
    ) -> pd.DataFrame:
        """
        Standardize and save extracted co-action data under co_action_data.

        :param df: Extracted co-action dataframe to standardize and save.
        :param filename: Name of the CSV file to save under co_action_data.
        :param object_column: Optional source column to rename to objectId.
        :param content_type: Optional contentType value to assign to every row.
        :return: Standardized co-action dataframe that was saved.
        """
        result_df = self.standardize_object_data(df, object_column=object_column, content_type=content_type)
        self.ch.save_dataframe(result_df, self.dm.path_co_action_data + filename)
        self.lm.printl(f"Number of extracted objects: {len(result_df)}, unique userIds: {len(result_df['userId'].unique())}")
        return result_df

    @log_method
    def extract_url_dataset(
        self,
        df: pd.DataFrame,
        filename: str,
        known_url: Sequence[str],
        parse_urls: bool = True
    ) -> pd.DataFrame:
        """
        Extract, parse, standardize, and save URL co-action objects.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the URL co-action CSV file to save under co_action_data.
        :param known_url: Domains that should not be unshortened.
        :param parse_urls: Whether to parse/unshorten URLs before saving.
        :return: Standardized URL co-action dataframe.
        """
        filter_df = self.extract_url_data(df)
        result_df = self.url_preprocessor.build_url_dataframe(filter_df, known_url, parse_urls=parse_urls)
        self.lm.printl(result_df.shape)
        self.lm.printl(result_df.columns)

        object_column = "domainUrl" if "domainUrl" in result_df.columns else "url"
        return self.save_standard_object_data(result_df, filename, object_column=object_column)

    @log_method
    def extract_hashtag_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract, standardize, and save hashtag co-action objects.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the hashtag co-action CSV file to save under co_action_data.
        :return: Standardized hashtag co-action dataframe.
        """
        filter_df = self.extract_hashtag_data(df)
        if filter_df.empty or "hashtag_list" not in filter_df.columns:
            return self.save_standard_object_data(filter_df, filename)
        result_df = self.content_preprocessor.build_list_object_dataframe(filter_df, "hashtag_list")
        return self.save_standard_object_data(result_df, filename, object_column="objectId")

    @log_method
    def extract_mention_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract, standardize, and save mention co-action objects.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the mention co-action CSV file to save under co_action_data.
        :return: Standardized mention co-action dataframe.
        """
        filter_df = self.extract_mention_data(df)
        if filter_df.empty or "mention_list" not in filter_df.columns:
            return self.save_standard_object_data(filter_df, filename)
        result_df = self.content_preprocessor.build_list_object_dataframe(filter_df, "mention_list")
        return self.save_standard_object_data(result_df, filename, object_column="objectId")

    @log_method
    def extract_retweet_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract, standardize, and save retweet co-action objects.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the retweet co-action CSV file to save under co_action_data.
        :return: Standardized retweet co-action dataframe.
        """
        return self.save_standard_object_data(self.extract_retweet_data(df), filename, object_column="objectId")

    @log_method
    def extract_reply_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract, standardize, and save reply co-action objects.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the reply co-action CSV file to save under co_action_data.
        :return: Standardized reply co-action dataframe.
        """
        return self.save_standard_object_data(self.extract_reply_data(df), filename, object_column="objectId")

    @log_method
    def extract_comment_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract, standardize, and save comment co-action objects.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the comment co-action CSV file to save under co_action_data.
        :return: Standardized comment co-action dataframe.
        """
        return self.save_standard_object_data(self.extract_comment_data(df), filename, object_column="objectId")

    @log_method
    def extract_text_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and standardize text co-action objects before embedding.

        :param df: Normalized dataset dataframe.
        :return: Standardized text co-action dataframe.
        """
        return self.standardize_object_data(self.extract_text_data(df), object_column="objectId")

    @log_method
    def filter_content_df(
        self,
        df: pd.DataFrame,
        column: str,
        excludeList: Sequence[str],
        filename: str
    ) -> pd.DataFrame:
        """
        Optionally remove excluded objects from an already extracted co-action dataframe.

        :param df: Co-action dataframe to filter.
        :param column: Name of the object column to compare with the exclusion list.
        :param excludeList: Values to remove from the dataframe.
        :param filename: Name of the filtered CSV file to save under co_action_data.
        :return: Filtered co-action dataframe.
        """
        self.lm.printl(
            f"{self.file_name}. Before filtering #rows: {df.shape[0]}, "
            f"# distinct {column}s: {len(df[column].unique())}"
        )

        normalized_exclude_list = [value.lower() for value in excludeList]
        result_df = df[~df[column].isin(normalized_exclude_list)]

        self.ch.save_dataframe(result_df, self.dm.path_co_action_data + filename)

        self.lm.printl(
            f"{self.file_name}. After filtering #rows: {result_df.shape[0]}, "
            f"# distinct {column}s: {len(result_df[column].unique())}"
        )
        return result_df
