import numpy as np
import pandas as pd

from .base_preprocessing import DatasetPreprocessing
from utils.decorator_definition import log_method


class MhPreprocessing(DatasetPreprocessing):
    """Preprocess the MH Reddit comment sample into the framework schema."""

    @log_method
    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize MH Reddit rows and save them under temp_data.

        :param df: Raw MH dataframe with Reddit post/comment columns.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized MH dataframe.
        """
        df = df.copy()
        df["is_comment"] = df["is_comment"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["id"] = np.where(df["is_comment"], df["comment_id"], df["post_id"])
        df["id"] = df["id"].fillna(df["comment_id"]).fillna(df["post_id"]).fillna(df["link_id"])
        df["userId"] = df["username"]
        df["created"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        df["created"] = df["created"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["contentType"] = np.where(df["is_comment"], "comment", "post")

        selected_columns = [
            "id",
            "userId",
            "created",
            "contentType",
            "comment_id",
            "link_id",
            "parent_id",
            "text",
            "label",
            "MH_category",
            "subreddit",
            "specific_disorder",
        ]
        df = df[selected_columns]
        df = df[df["id"].notna() & df["userId"].notna()]
        df = df.sort_values(by=["created"])
        df = df.reset_index(drop=True)
        df = self.standardize_normalized_data(df)

        self.ch.save_dataframe(df, self.dm.path_temp_data + filename)
        return df

    def extract_comment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Reddit submission ids as co-comment objects.

        Two users co-comment when they both comment under the same Reddit post. The
        post/thread id is stored in link_id; parent_id is intentionally ignored so
        nested reply structure is collapsed to the post level.

        :param df: Normalized MH dataframe.
        :return: Dataframe with post ids renamed to objectId.
        """
        filter_df = df[df["comment_id"].notna() & df["link_id"].notna()][
            ["id", "userId", "created", "link_id", "contentType"]
        ].copy()
        return filter_df.rename(columns={"link_id": "objectId"})

    def extract_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Reddit text as objectId values for text similarity.

        :param df: Normalized MH dataframe.
        :return: Dataframe with text values renamed to objectId.
        """
        filter_df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")][
            ["id", "userId", "created", "text", "contentType"]
        ].copy()
        return filter_df.rename(columns={"text": "objectId"})

    def extract_url_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty URL dataframe because MH does not configure a URL layer.

        :param df: Normalized MH dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_hashtag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty hashtag dataframe because MH does not configure a hashtag layer.

        :param df: Normalized MH dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_mention_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty mention dataframe because MH does not configure a mention layer.

        :param df: Normalized MH dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_retweet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty retweet dataframe because MH is a Reddit dataset.

        :param df: Normalized MH dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_reply_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty reply dataframe because MH currently uses only co-comment.

        :param df: Normalized MH dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()
