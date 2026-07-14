import pandas as pd

from .base_preprocessing import DatasetPreprocessing
from utils.decorator_definition import log_method


class MoltbookPreprocessing(DatasetPreprocessing):
    """Preprocess Moltbook comments and posts into the framework schema."""

    @log_method
    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize a Moltbook comment or post file and save it under temp_data.

        :param df: Raw Moltbook comment or post dataframe.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized Moltbook dataframe.
        """
        if 'comment' in filename:
            self.lm.printl(f"{self.file_name}. normalize_data_moltbook comments.")
            rename_dict = {
                'created_at': 'created',
                'author_id': 'userId',
                'post_id': 'replyId',
                'content': 'text'
            }

            df['contentType'] = 'reply'
            selected_column = ['id', 'created', 'userId', 'text', 'replyId', 'contentType']
        elif 'post' in filename:
            self.lm.printl(f"{self.file_name}. normalize_data_moltbook posts.")
            rename_dict = {
                'created_at': 'created',
                'author_id': 'userId',
                'content': 'text'
            }
            selected_column = ['id', 'created', 'userId', 'text', 'contentType']
            df['contentType'] = 'original'
        else:
            raise ValueError("Moltbook normalization expects 'comment' or 'post' in filename.")

        df = df.rename(columns=rename_dict)
        df = df[selected_column]

        df['created'] = pd.to_datetime(df['created'], format='ISO8601', utc=True)
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df = df.sort_values(by=['created'])
        df = df.reset_index(drop=True)

        df = self.standardize_normalized_data(df)
        self.ch.save_dataframe(df, self.dm.path_temp_data + filename)

        return df

    def extract_url_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract URL lists from Moltbook text content.

        :param df: Normalized Moltbook dataframe containing a text column.
        :return: Dataframe with URL lists ready for common URL processing.
        """
        df = df.copy()
        df["url_list"] = df["text"].apply(self.url_preprocessor.extract_urls)

        return df[['id', 'userId', 'created', 'url_list', 'contentType']].copy()

    def extract_hashtag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty hashtag dataframe because Moltbook has no hashtag layer.

        :param df: Normalized Moltbook dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_mention_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty mention dataframe because Moltbook has no mention layer.

        :param df: Normalized Moltbook dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_retweet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return an empty retweet dataframe because Moltbook has no retweet layer.

        :param df: Normalized Moltbook dataframe, unused for this dataset.
        :return: Empty standardized co-action dataframe.
        """
        return self.empty_standard_dataframe()

    def extract_reply_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comment-to-post reply objects from Moltbook comments.

        :param df: Normalized Moltbook comment dataframe.
        :return: Dataframe with reply targets renamed to objectId.
        """
        if 'replyId' not in df.columns:
            return self.empty_standard_dataframe()

        filter_df = df[df['replyId'].isnull() == False][
            ['id', 'userId', 'created', 'replyId', 'contentType']
        ].copy()
        return filter_df.rename(columns={'replyId': 'objectId'})

    def extract_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Moltbook text as objectId values for text similarity.

        :param df: Normalized Moltbook dataframe containing text content.
        :return: Dataframe with text values renamed to objectId.
        """
        filter_df = df[df['text'].isnull() == False][
            ['id', 'userId', 'created', 'text', 'contentType']
        ].copy()
        return filter_df.rename(columns={'text': 'objectId'})
