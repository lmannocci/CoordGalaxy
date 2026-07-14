from DirectoryManager import DirectoryManager
from utils.Checkpoint.Checkpoint import *
from InputManager.dataset import DatasetPreprocessing, build_dataset_preprocessor
from utils.decorator_definition import *

import os

import pandas as pd

file_name = os.path.splitext(os.path.basename(__file__))[0]
absolute_path = os.path.dirname(__file__)
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")

class InputManager:
    """Facade for dataset input normalization, extraction, and optional filtering."""

    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dataset_name: str) -> None:
        """
        Create an input manager for a configured dataset.

        :param dataset_name: Name of the dataset registered in the dataset preprocessor factory.
        :return: None.
        """
        self.lm = LogManager("main")

        self.ch = Checkpoint()
        self.dataset_name = dataset_name
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path)

    def _build_dataset_preprocessor(self) -> DatasetPreprocessing:
        """
        Build the dataset-specific preprocessor registered for this dataset.

        :return: DatasetPreprocessing instance for the current dataset.
        """
        return build_dataset_preprocessor(self.dataset_name, self.ch, self.dm, self.lm, file_name)

    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------

    @log_method
    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
            Normalize data, performing some changes in the dataframe, e.g., renaming columns or changing columns' type.
            :param df: [DataFrame] DataFrame on which performing operations.
            :param filename: [str] Filename of a csv file.
            :return: [DataFrame] Return a pandas DataFrame with the performed changes
        """
        return self._build_dataset_preprocessor().normalize_data(df, filename)
    
    @log_method
    def normalize_data_text(self, df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
        """
            Normalize data text, performing some changes in the dataframe, e.g., renaming columns or changing columns' type.
            :param df: [DataFrame] DataFrame on which performing operations.
            :param filename: [str] Filename of a csv file.
            :return: [DataFrame | None] Return the normalized text dataframe, or None when the dataset has no text file.
        """
        return self._build_dataset_preprocessor().normalize_data_text(df, filename)

   
    @log_method
    def normalize_data_user(self, df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
        """
            Normalize data of the users, performing some changes in the dataframe, e.g., renaming columns or changing columns' type.
            :param df: [DataFrame] DataFrame on which performing operations.
            :param filename: [str] Filename of a csv file.
            :return: [DataFrame | None] Return the normalized users dataframe, or None when the dataset has no users file.
        """
        return self._build_dataset_preprocessor().normalize_user_data(df, filename)

    @log_method
    def extract_url_dataset(self, df: pd.DataFrame, filename: str, known_url: list[str], parse_urls: bool = True) -> pd.DataFrame:
        
        """
            Given a dataframe of posts published by several users, it extracts URLs of original tweets (retweet and reply excluded)
             contained in each post.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :param known_url: [list] List of domains that should not be unshortened.
            :param parse_urls: [bool] Whether to parse/unshorten URL values before saving.
            :return: [DataFrame] Return a pandas DataFrame with extracted URLs.
        """
        return self._build_dataset_preprocessor().extract_url_dataset(df, filename, known_url, parse_urls=parse_urls)

    @log_method
    def extract_text_dataset(self, df: pd.DataFrame, filename: str, build_embeddings: bool = True) -> pd.DataFrame:
        """
        Extract text objects and save both the text co-action CSV and embedding NPY file.

        :param df: Normalized dataset dataframe containing text content.
        :param filename: Name of the NPY embeddings file to save under co_action_data.
        :param build_embeddings: Whether to build and save embeddings in the NPY file.
        :return: Standardized text co-action dataframe.
        """
        text_df = self._build_dataset_preprocessor().extract_text_dataset(df)
        csv_filename = os.path.splitext(filename)[0] + ".csv"
        self.ch.save_dataframe(text_df, f"{self.dm.path_co_action_data}{csv_filename}")
        if build_embeddings:
            from InputManager.Content.embedding import EmbeddingManager

            em = EmbeddingManager(self.dataset_name, self.ch, self.lm)
            em.build_multilingual_embeddings(text_df, text_col="objectId", output_path=f"{self.dm.path_co_action_data}{filename}", model_name="intfloat/multilingual-e5-large", batch_size=32)
        return text_df

    @log_method
    def extract_comment_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract comment co-action objects for datasets that support them.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the comment co-action CSV file to save under co_action_data.
        :return: Standardized comment co-action dataframe.
        """
        return self._build_dataset_preprocessor().extract_comment_dataset(df, filename)

    @log_method
    def filter_content_df(
        self,
        df: pd.DataFrame,
        column: str,
        excludeList: list[str],
        filename: str
    ) -> pd.DataFrame:
        """
        Optionally filter extracted co-action objects with a shared exclusion list.

        :param df: Co-action dataframe to filter.
        :param column: Name of the object column to compare with the exclusion list.
        :param excludeList: Values to remove from the dataframe.
        :param filename: Name of the filtered CSV file to save under co_action_data.
        :return: Filtered co-action dataframe.
        """
        return self._build_dataset_preprocessor().filter_content_df(df, column, excludeList, filename)


    @log_method
    def extract_hashtag_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
            Given a dataframe of posts published by several users, it extracts hashtags of original tweets (retweet and reply excluded)
             contained in each post.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :return: [DataFrame] Return a pandas DataFrame with extracted hashtags.
        """
        return self._build_dataset_preprocessor().extract_hashtag_dataset(df, filename)


    @log_method
    def extract_mention_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
            Given a dataframe of posts published by several users, it extracts mentions of original tweets (retweet and reply excluded)
             contained in each post.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :return: [DataFrame] Return a pandas DataFrame with extracted mentions.
        """
        return self._build_dataset_preprocessor().extract_mention_dataset(df, filename)

    @log_method
    def extract_retweet_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract retweet co-action objects for datasets that support them.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the retweet co-action CSV file to save under co_action_data.
        :return: Standardized retweet co-action dataframe.
        """
        return self._build_dataset_preprocessor().extract_retweet_dataset(df, filename)

    @log_method
    def extract_reply_dataset(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Extract reply co-action objects for datasets that support them.

        :param df: Normalized dataset dataframe.
        :param filename: Name of the reply co-action CSV file to save under co_action_data.
        :return: Standardized reply co-action dataframe.
        """
        return self._build_dataset_preprocessor().extract_reply_dataset(df, filename)
