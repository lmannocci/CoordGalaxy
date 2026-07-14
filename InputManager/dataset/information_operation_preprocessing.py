import numpy as np
import pandas as pd

from .base_preprocessing import DatasetPreprocessing
from utils.decorator_definition import log_method


class InformationOperationPreprocessing(DatasetPreprocessing):
    """Preprocess datasets from the Information Operations collection."""

    @log_method
    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize Information Operations post data and save it under temp_data.

        :param df: Raw Information Operations post dataframe.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized Information Operations dataframe.
        """
        # already filtered by language during notebook preprocessing
        rename_dict = {
            'postid': 'id',
            'post_time': 'created',
            'accountid': 'userId',
            'hashtags': 'hashtag_list',
            'urls': 'url_list',
            'account_mentions': 'mention_list',
            'reposted_accountid': 'retweetId',
            'in_reply_to_accountid': 'replyId',
            'is_control': 'isControl'
        }
        df = df.rename(columns=rename_dict)

        selected_column = ['id', 'created', 'userId', 'hashtag_list', 'url_list',
                           'mention_list', 'retweetId', 'replyId', 'isControl']
        df = df[selected_column]

        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['contentType'] = np.where(
            df['retweetId'].isnull() == False,
            'retweet',
            np.where(df['replyId'].isnull() == False, "reply", "original")
        )

        df = df.sort_values(by=['created'])
        df = df.reset_index(drop=True)
        df = self.standardize_normalized_data(df)

        self.ch.save_dataframe(df, self.dm.path_temp_data + filename)

        return df

    def extract_url_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract URL lists from Information Operations normalized rows.

        :param df: Normalized Information Operations dataframe.
        :return: Dataframe with URL lists ready for common URL processing.
        """
        not_nan_url = df[
            df['url_list'].apply(
                lambda value: (
                    isinstance(value, list) and len(value) > 0
                ) or (
                    isinstance(value, str) and value.strip() not in ['[]', '']
                )
            )
        ]

        filter_df = not_nan_url[['id', 'userId', 'created', 'url_list', 'contentType', 'isControl']].copy()
        filter_df['url_list'] = filter_df['url_list'].apply(self.url_preprocessor.fix_and_parse_urls)
        return filter_df

    def extract_hashtag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hashtag lists from Information Operations normalized rows.

        :param df: Normalized Information Operations dataframe.
        :return: Dataframe with hashtag lists ready for common list processing.
        """
        not_nan_hashtag = df[df['hashtag_list'].apply(self.content_preprocessor.has_content_list)]
        filter_df = not_nan_hashtag[
            ['id', 'userId', 'created', 'hashtag_list', 'contentType', 'isControl']
        ].copy()
        filter_df['hashtag_list'] = filter_df['hashtag_list'].apply(self.content_preprocessor.ensure_list)
        return filter_df

    def extract_mention_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract mention lists from Information Operations normalized rows.

        :param df: Normalized Information Operations dataframe.
        :return: Dataframe with mention lists ready for common list processing.
        """
        not_nan_mention = df[df['mention_list'].apply(self.content_preprocessor.has_content_list)]
        filter_df = not_nan_mention[
            ['id', 'userId', 'created', 'mention_list', 'contentType', 'isControl']
        ].copy()
        filter_df['mention_list'] = filter_df['mention_list'].apply(self.content_preprocessor.ensure_list)
        return filter_df

    def extract_retweet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract retweeted account objects and map them with the shared id mapper.

        :param df: Normalized Information Operations dataframe.
        :return: Dataframe with retweeted account ids in objectId.
        """
        filter_df = df[df['retweetId'].isnull() == False][
            ['id', 'userId', 'created', 'retweetId', 'contentType', 'isControl']
        ].copy()
        filter_df = filter_df.rename(columns={'retweetId': 'objectId'})
        filter_df['objectId'] = self.user_id_mapper.map_series(filter_df['objectId'])
        return filter_df

    def extract_reply_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract replied account objects and map them with the shared id mapper.

        :param df: Normalized Information Operations dataframe.
        :return: Dataframe with replied account ids in objectId.
        """
        filter_df = df[df['replyId'].isnull() == False][
            ['id', 'userId', 'created', 'replyId', 'contentType', 'isControl']
        ].copy()
        filter_df = filter_df.rename(columns={'replyId': 'objectId'})
        filter_df['objectId'] = self.user_id_mapper.map_series(filter_df['objectId'])
        return filter_df

    def extract_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract text objects when Information Operations text is available.

        :param df: Normalized Information Operations dataframe.
        :return: Dataframe with text values renamed to objectId, or an empty standard dataframe.
        """
        if 'text' not in df.columns:
            return self.empty_standard_dataframe(include_is_control=True)

        filter_df = df[df['text'].isnull() == False][
            ['id', 'userId', 'created', 'text', 'contentType', 'isControl']
        ].copy()
        return filter_df.rename(columns={'text': 'objectId'})
