import numpy as np
import pandas as pd
from ast import literal_eval

from .base_preprocessing import DatasetPreprocessing
from utils.decorator_definition import log_method


class Uk2019Preprocessing(DatasetPreprocessing):
    """Preprocess UK 2019 tweet and user files into the framework schema."""

    @log_method
    def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize UK tweet metadata and save it under temp_data.

        :param df: Raw UK tweet dataframe.
        :param filename: Name of the normalized CSV file to save under temp_data.
        :return: Normalized UK tweet dataframe.
        """
        rename_dict = {
            "in_reply_to_status_id_str": "replyId",
            "id_str": "id",
            "created_at": "created",
            "in_reply_to_user_id_str": "replyUserId",
            "entities.urls": "urls",
            "entities.hashtags": "hashtags",
            "entities.user_mentions": "mentions",
            "retweeted_status.extended_tweet.entities.urls": "retweetExtendedUrls",
            "retweeted_status.extended_tweet.entities.hashtags": "retweetExtendedHashtags",
            "retweeted_status.extended_tweet.entities.user_mentions": "retweetExtendedMentions",
            "retweeted_status.id_str": "retweetId",
            "retweeted_status.created_at": "retweetCreated",
            "retweeted_status.user.id_str": "retweetUserId",
            'retweeted_status.entities.urls': "retweetUrls",
            'retweeted_status.entities.hashtags': "retweetHashtags",
            'retweeted_status.entities.user_mentions': "retweetMentions",
            "user.id_str": "userId",
            'extended_tweet.entities.user_mentions': "extendedMentions",
            'extended_tweet.entities.hashtags': "extendedHashtags",
            'extended_tweet.entities.urls': "extendedUrls"
        }
        df = df.rename(columns=rename_dict)

        df['created'] = pd.to_datetime(df['created'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['retweetCreated'] = pd.to_datetime(df['retweetCreated'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['retweetCreated'] = df['retweetCreated'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['contentType'] = np.where(
            df['retweetId'].isnull() == False,
            'retweet',
            np.where(df['replyId'].isnull() == False, "reply", "original")
        )

        df = df.sort_values(by=['created'])
        df = self.standardize_normalized_data(df)
        self.ch.save_dataframe(df, self.dm.path_temp_data + filename)

        return df

    @log_method
    def normalize_user_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize UK user metadata when a users file is available.

        :param df: Raw UK users dataframe.
        :param filename: Name of the normalized users CSV file to save under temp_data.
        :return: Normalized UK users dataframe.
        """
        df.drop(columns=['_index', '_type', '_id', '_score'], inplace=True, errors="ignore")
        rename_dict = {
            'id_str': 'userId',
            'followers_count': 'nFollowers',
            'created_at': 'created',
            'friends_count': 'nFriends',
            'favourites_count': 'nLikedTweets',
            'statuses_count': 'nPostedTweets',
            'screen_name': 'screenName',
            'botometer.scores.sentiment': 'botScoreSentiment',
            'botometer.scores.english': 'botScoreEnglish',
            'botometer.scores.friend': 'botScoreFriend',
            'botometer.scores.universal': 'botScoreUniversal',
            'botometer.scores.user': 'botScoreUser',
            'botometer.scores.content': 'botScoreContent',
            'botometer.scores.temporal': 'botScoreTemporal',
            'botometer.scores.network': 'botScoreNetwork',
            'botometer.is_bot': 'isBot',
            'botometer.is_bot_english': 'isBotEnglish',
            'botometer.skipped': 'botSkipped'
        }
        df = df.rename(columns=rename_dict)
        df['created'] = pd.to_datetime(df['created'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        columns = ['userId', 'name', 'screenName', 'nFollowers', 'nFriends', 'description', 'created',
                   'nLikedTweets', 'nPostedTweets', 'location',
                   'botScoreSentiment', 'botScoreEnglish', 'botScoreFriend',
                   'botScoreUniversal', 'botScoreUser', 'botScoreContent',
                   'botScoreTemporal', 'botScoreNetwork', 'isBot', 'isBotEnglish',
                   'botSkipped']
        df = df[columns]
        df = self.map_user_ids(df)

        self.ch.save_dataframe(df, self.dm.path_temp_data + filename)
        return df

    @log_method
    def normalize_data_text(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Normalize UK text data when a separate text file is available.

        :param df: Raw UK text dataframe.
        :param filename: Name of the normalized text CSV file to save under temp_data.
        :return: Normalized UK text dataframe.
        """
        rename_dict = {
            "id_str": "id",
            "created_at": "created",
            "user.id_str": "userId",
        }
        df = df.rename(columns=rename_dict)
        df = df[df['created'].isnull() == False]
        df = df[df['created'] != '0']

        df['created'] = pd.to_datetime(df['created'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['contentType'] = np.where(
            df['retweetId'].isnull() == False,
            'retweet',
            np.where(df['replyId'].isnull() == False, "reply", "original")
        )
        df = df.sort_values(by=['created'])
        df = df[
            ['id', 'userId', 'created', 'text', 'retweet_count', 'favorite_count',
             'reply_count', 'quote_count', 'contentType']
        ]
        df = self.standardize_normalized_data(df)
        self.ch.save_dataframe(df, self.dm.path_temp_data + filename)
        return df

    def extract_url_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract URL lists from UK tweet entities.

        :param df: Normalized UK tweet dataframe.
        :return: Dataframe with URL lists ready for common URL processing.
        """
        not_nan_url = df[df['urls'].isnull() == False].copy()
        not_nan_url['url_list'] = not_nan_url['urls'].apply(literal_eval)
        not_nan_url['url_list'] = not_nan_url['url_list'].apply(
            self.extract_element,
            column='expanded_url'
        )

        return not_nan_url[['id', 'userId', 'created', 'url_list', 'contentType']].copy()

    def extract_element(self, original_list: list[dict], column: str) -> list:
        """
        Extract one field from every entity dictionary in a list.

        :param original_list: List of entity dictionaries.
        :param column: Entity field to extract.
        :return: List of extracted field values.
        """
        return [element[column] for element in original_list]

    def extract_hashtag_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract hashtag lists from UK tweet entities.

        :param df: Normalized UK tweet dataframe.
        :return: Dataframe with hashtag lists ready for common list processing.
        """
        not_nan_hashtag = df[df['hashtags'].isnull() == False].copy()
        not_nan_hashtag['hashtag_list'] = not_nan_hashtag['hashtags'].apply(literal_eval)
        not_nan_hashtag['hashtag_list'] = not_nan_hashtag['hashtag_list'].apply(
            self.extract_element,
            column='text'
        )
        return not_nan_hashtag[['id', 'userId', 'created', 'hashtag_list', 'contentType']].copy()

    def extract_mention_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract mention id lists from UK tweet entities.

        :param df: Normalized UK tweet dataframe.
        :return: Dataframe with mention id lists ready for common list processing.
        """
        not_nan_mention = df[df['mentions'].isnull() == False].copy()
        not_nan_mention['mention_list'] = not_nan_mention['mentions'].apply(literal_eval)
        not_nan_mention['mention_list'] = not_nan_mention['mention_list'].apply(
            self.extract_element,
            column='id'
        )
        return not_nan_mention[['id', 'userId', 'created', 'mention_list', 'contentType']].copy()

    def extract_retweet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract retweet status ids from UK normalized tweets.

        :param df: Normalized UK tweet dataframe.
        :return: Dataframe with retweet status ids in objectId.
        """
        filter_df = df[df['retweetId'].isnull() == False][
            ['id', 'userId', 'created', 'retweetId', 'contentType']
        ].copy()
        return filter_df.rename(columns={'retweetId': 'objectId'})

    def extract_reply_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract reply status ids from UK normalized tweets.

        :param df: Normalized UK tweet dataframe.
        :return: Dataframe with reply status ids in objectId.
        """
        filter_df = df[df['replyId'].isnull() == False][
            ['id', 'userId', 'created', 'replyId', 'contentType']
        ].copy()
        return filter_df.rename(columns={'replyId': 'objectId'})

    def extract_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract tweet text as objectId values for text similarity.

        :param df: Normalized UK tweet dataframe.
        :return: Dataframe with text values renamed to objectId, or an empty standard dataframe.
        """
        if 'text' not in df.columns:
            return self.empty_standard_dataframe()

        filter_df = df[df['text'].isnull() == False][
            ['id', 'userId', 'created', 'text', 'contentType']
        ].copy()
        return filter_df.rename(columns={'text': 'objectId'})
