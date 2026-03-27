from IntegrityConstraintManager.IntegrityConstraintManager import *
from DirectoryManager import DirectoryManager
from utils.ElasticSearch import *
from utils.Checkpoint.Checkpoint import *
from InputManager.Content.embedding import EmbeddingManager

import elasticsearch
import json
from jinja2 import Template
import pandas as pd
from pandasticsearch import Select
import numpy as np
import swifter
import ast
import re


# Packages to extract URLs
from ast import literal_eval
from unshortenit import UnshortenIt
from urllib.parse import urlparse
from requests.exceptions import ConnectionError

file_name = os.path.splitext(os.path.basename(__file__))[0]
absolute_path = os.path.dirname(__file__)
es_query_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}utils{os.sep}ElasticSearch{os.sep}queries{os.sep}")
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")

COUNTER_UNSHORTEN = 0
COUNTER_PARSING = 1

class InputManager:

    # PRIVATE METHODS
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dataset_name):
        self.lm = LogManager("main")

        self.ch = Checkpoint()
        self.dataset_name = dataset_name
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path)

    def __searchElasticByDate(self, startDate, endDate):
        """
            Retrieve data from Elasticsearch of a specific period between the specified dates.
            :param startDate: [str] Date of the start period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param endDate: [str] Date of the end period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :return: [DataFrame] Return a pandas DataFrame of the json data. An empty dataset is return in case no data are retrieved from Elasticsearch
        """
        if elasticsearch.__version__[0] >= 8:
            with open(es_query_path + "date_query_new.json") as file:
                template = Template(file.read())
                query = json.loads(template.render(startDate=startDate, endDate=endDate))
        else:
            with open(es_query_path + "date_query.json") as file:
                template = Template(file.read())
                query = template.render(startDate=startDate, endDate=endDate)

        resp = self.ind.searchQuery(query)
        hits = len(resp['hits']['hits'])
        self.lm.printl(f"{file_name}: searchElasticByDate - Number of results retrieved from es: " + str(
            hits) + " -> startDate: " + startDate + " endDate: " + endDate)
        if hits > 0:
            df = Select.from_dict(resp).to_pandas()
            return df
        else:
            return pd.DataFrame()

    def __searchElasticAll(self):
        """
            Retrieve all data from Elasticsearch.
            :return: [DataFrame] Return a pandas DataFrame of the json data. An empty dataset is return in case no data are retrieved from Elasticsearch
        """
        if elasticsearch.__version__[0] >= 8:
            with open(es_query_path + "all_query_new.json") as file:
                template = Template(file.read())
                query = json.loads(template.render())
        else:
            with open(es_query_path + "all_query.json") as file:
                template = Template(file.read())
                query = template.render()

        resp = self.ind.searchQuery(query)
        hits = len(resp['hits']['hits'])
        self.lm.printl(
            f"{file_name}: Data retrieved from Elasticsearch. username_index: " + self.username_index + ", index_name: " + self.index_name + ". Number of results retrieved from es: " + str(
                hits))
        if hits > 0:
            df = Select.from_dict(resp).to_pandas()
            return df
        else:
            return pd.DataFrame()

    def __searchElasticSourcesUsersByDate(self, startDate, endDate, sourceList, userList, filename):
        """
            Retrieve data, selecting fields in sources from Elasticsearch of a specific period between the specified dates
            related at the users specified in userList
            :param startDate: [str] Date of the start period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param endDate: [str] Date of the end period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param sourceList: [list] List of sources fields to retrieve among index fields
            :param sourceList: [list] List of userIds for which retrieve the documents
        """

        sourceListStr = '","'.join(sourceList)
        sourceListStr = '["' + sourceListStr + '"]'
        userListStr = '","'.join(userList)
        userListStr = '["' + userListStr + '"]'

        if elasticsearch.__version__[0] >= 8:
            with open(es_query_path + "date_query_sources_users_new.json") as file:
                template = Template(file.read())
                query = json.loads(template.render(startDate=startDate, endDate=endDate, sourceList=sourceListStr, userList=userListStr))
        else:
            with open(es_query_path + "date_query_sources.json") as file:
                template = Template(file.read())
                query = template.render(startDate=startDate, endDate=endDate, sourceList=sourceListStr, userList=userListStr)

        self.__save_ES_yielded_data(query, filename, startDate=startDate, endDate=endDate, sourceListStr=sourceListStr)

    def __searchElasticSources(self, sourceList, filename):
        """
            Retrieve data, selecting fields in sources from Elasticsearch of a specific period between the specified dates.
            :param startDate: [str] Date of the start period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param endDate: [str] Date of the end period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param sourceList: [list] List of sources fields to retrieve among index fields
        """

        sourceListStr = '","'.join(sourceList)
        sourceListStr = '["' + sourceListStr + '"]'

        if elasticsearch.__version__[0] >= 8:
            with open(es_query_path + "query_sources_new.json") as file:
                template = Template(file.read())
                query = json.loads(template.render(sourceList=sourceListStr))
        else:
            with open(es_query_path + "query_sources.json") as file:
                template = Template(file.read())
                query = template.render(sourceList=sourceListStr)
        resp = self.ind.searchQuery(query)
        hits = len(resp['hits']['hits'])
        self.lm.printl(
            f"{file_name}: Data retrieved from Elasticsearch. username_index: " + self.username_index + ", index_name: " + self.index_name + ". Number of results retrieved from es: " + str(
                hits))
        if hits > 0:
            df = Select.from_dict(resp).to_pandas()
            self.ch.save_dataframe(df, self.dm.path_dataset + filename)
            return df
        else:
            return pd.DataFrame()

        # self.__save_ES_yielded_data(query, filename, startDate=startDate, endDate=endDate, sourceListStr=sourceListStr)

    def __searchElasticSourcesByDate(self, startDate, endDate, sourceList, filename):
        """
            Retrieve data, selecting fields in sources from Elasticsearch of a specific period between the specified dates.
            :param startDate: [str] Date of the start period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param endDate: [str] Date of the end period from which retrieving data. Format: '%Y-%m-%d %H:%M:%S'
            :param sourceList: [list] List of sources fields to retrieve among index fields
        """

        sourceListStr = '","'.join(sourceList)
        sourceListStr = '["' + sourceListStr + '"]'

        if elasticsearch.__version__[0] >= 8:
            with open(es_query_path + "date_query_sources_new.json") as file:
                template = Template(file.read())
                query = json.loads(template.render(startDate=startDate, endDate=endDate, sourceList=sourceListStr))
        else:
            with open(es_query_path + "date_query_sources.json") as file:
                template = Template(file.read())
                query = template.render(startDate=startDate, endDate=endDate, sourceList=sourceListStr)

        self.__save_ES_yielded_data(query, filename, startDate=startDate, endDate=endDate, sourceListStr=sourceListStr)

    def __save_ES_yielded_data(self, query, filename, startDate=None, endDate=None, sourceListStr=None):
        resp = self.ind.searchQueryYield(query)
        # hits = len(resp['hits']['hits'])

        self.lm.printl(f"{file_name}. Start writing JSON file: " + self.dm.path_dataset + filename)
        hits = 0

        with open(self.dm.path_dataset + filename, 'w') as file:
            for row in resp:
                hits += 1
                self.lm.printK(hits, 1000, "InputManager. Write row: " + str(hits))
                file.write(json.dumps(row['_source']))
                file.write("\n")

        self.lm.printl(f"{file_name} searchElasticSourcesByDate - Number of results retrieved from es: " + str(hits) +
                       " -> startDate: " + startDate + " endDate: " + endDate + "sourceList: " + sourceListStr)

        self.lm.printl(f"{file_name}. Finished writing JSON file: " + self.dm.path_dataset + filename)


    # def __process_url_apply(self, row):
    #     self.lm.printK(row.name, 1000, f"Processing url {str(row.name)}.")
    #     url = row['url']
    #     unshortener = UnshortenIt()
    #     try:
    #         uri = unshortener.unshorten(url)
    #     except ConnectionError as e:
    #         # self.lm.printl(f"ConnectionError: {e}.")
    #         uri = url
    #     except Exception as e:
    #         # self.lm.printl(f"Unshortener error: {e}. URL: {url}")
    #         uri = url
    #
    #     try:
    #         parsed_url = urlparse(uri)
    #         # Extracting different components
    #         #         row['scheme'] = parsed_url.scheme
    #         #         row['netloc'] = parsed_url.netloc
    #         domainUrl = parsed_url.netloc.replace('www.', '')
    #         pathUrl = parsed_url.path
    #     #         row['params'] = parsed_url.params
    #     #         row['query'] = parsed_url.query
    #     #         row['fragment'] = parsed_url.fragment
    #     except Exception as e:
    #         self.lm.printl(f"URLparse error: {e}. URI: {uri}")
    #         return pd.Series([np.nan, np.nan, np.nan])
    #     return pd.Series([uri, domainUrl, pathUrl])

    def __unshorten_url(self, url):
        global COUNTER_UNSHORTEN
        self.lm.printK(COUNTER_UNSHORTEN, 1000, f"Unshortening url {str(COUNTER_UNSHORTEN)}.")
        COUNTER_UNSHORTEN += 1
        unshortener = UnshortenIt()
        try:
            solved_url = unshortener.unshorten(url)
        except ConnectionError as e:
            # self.lm.printl(f"ConnectionError: {e}.")
            solved_url = url
        except Exception as e:
            # self.lm.printl(f"Unshortener error: {e}. URL: {url}")
            solved_url = url
        return solved_url

    def __parse_url(self, resolved_url):
        global COUNTER_PARSING
        self.lm.printK(COUNTER_PARSING, 100000, f"Parsing url {str(COUNTER_PARSING)}.")
        COUNTER_PARSING += 1
        try:
            parsed_url = urlparse(resolved_url)
            # Extracting different components
            #         row['scheme'] = parsed_url.scheme
            #         row['netloc'] = parsed_url.netloc
            domainUrl = parsed_url.netloc.replace('www.', '')
            pathUrl = parsed_url.path
        #         row['params'] = parsed_url.params
        #         row['query'] = parsed_url.query
        #         row['fragment'] = parsed_url.fragment
        except Exception as e:
            # print(f"URLparse error: {e}. URI: {resolved_url}")
            # return pd.Series([np.nan, np.nan, np.nan])
            return np.nan, np.nan
        # return pd.Series([uri, domainUrl, pathUrl])
        return domainUrl, pathUrl

    def __extract_element(self, original_list, c):
        mod_list = []
        for elem in original_list:
            mod_list.append(elem[c])
        return mod_list

    def __extract_retweet_reply(self, df, filename, c_original, c_mapped):
        """
            Given a dataframe of posts published by several users, it extracts retweet or reply.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :return: [DataFrame] Return a pandas DataFrame with extracted retweets or replies.
        """
        self.lm.printl(f"{file_name}. __extract_retweet_reply start.")

        df_not_null = df[df[c_original].isnull() == False]
        if self.dataset_name == "uk2019":
            filter_df = df_not_null[['id', 'userId', 'created', c_original, 'type']].copy()
        elif self.dataset_name == "ira":
            filter_df = df_not_null[['id', 'userId', 'created', c_original, 'type', 'class']].copy()
        elif self.dataset_name == "IORussia":
            filter_df = df_not_null[['id', 'userId', 'created', c_original, 'type', 'isControl']].copy()
        elif self.dataset_name == "moltbook":
            filter_df = df_not_null[['id', 'userId', 'created', c_original, 'type']].copy() 
        filter_df = filter_df.rename(columns={c_original: c_mapped})

        self.ch.save_dataframe(filter_df, self.dm.path_dataset + filename)

        self.lm.printl(
            f"Number of extracted retweets: {len(filter_df)}, unique userIds: {len(filter_df['userId'].unique())}")
        self.lm.printl(f"{file_name}. __extract_retweet_reply finish.")
        return filter_df

    def __normalize_data_ira(self, df, filename):
        df = df[df[' tweet lang'] == 'en']

        df = df.drop(columns = ['name', '# of likes', 'account lang', ' tweet lang', 
                                        '# of likes', '# of retweets', 
                                        'images', 'image hashes', 'length'])

        rename_dict = {
                    "tweet id": "id",
                    "tweet time": "created",
                    "user id": "userId",
                    "urls": "url_list",
                    "mentions": "mention_list",
                    "hashtags": "hashtag_list"
                }
        df = df.rename(columns=rename_dict)

        # convert datetime column to the best format
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['type'] =  "original"

        df = df.sort_values(by=['created'])
        df = df.reset_index(drop=True)

        self.ch.save_dataframe(df, self.dm.path_dataset + filename)
        self.lm.printl(f"{file_name}. normalize_data finish.")

        return df

    def __normalize_data_IORussia(self, df, filename):
        # already filtered by language during notebook preprocessing
        # df = df[df[' tweet lang'] == 'en']

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

        selected_column = ['id', 'created', 'userId', 'hashtag_list', 'url_list', 'mention_list', 'retweetId', 'replyId', 'isControl']
        df = df[selected_column]

        # convert datetime column to the best format
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['type'] = np.where(df['retweetId'].isnull() == False, 'retweet',
                              np.where(df['replyId'].isnull() == False, "reply", "original"))


        df = df.sort_values(by=['created'])
        df = df.reset_index(drop=True)

        self.ch.save_dataframe(df, self.dm.path_dataset + filename)
        self.lm.printl(f"{file_name}. normalize_data finish.")

        return df

    def __normalize_data_uk2019(self, df, filename):
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

        # convert datetime column to the best format
        df['created'] = pd.to_datetime(df['created'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['retweetCreated'] = pd.to_datetime(df['retweetCreated'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['retweetCreated'] = df['retweetCreated'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['type'] = np.where(df['retweetId'].isnull() == False, 'retweet',
                              np.where(df['replyId'].isnull() == False, "reply", "original"))

        df = df.sort_values(by=['created'])
        self.ch.save_dataframe(df, self.dm.path_dataset + filename)
        self.lm.printl(f"{file_name}. normalize_data finish.")

        return df
    
    def __normalize_data_moltbook(self, df, filename):
        rename_dict = {
                'created_at': 'created',
                'author_id': 'userId',
                'post_id': 'replyId'
            }
        df = df.rename(columns=rename_dict)

        selected_column = ['id', 'created', 'userId', 'hashtag_list', 'url_list', 'mention_list', 'retweetId', 'replyId', 'isControl']
        df = df[selected_column]

        # convert datetime column to the best format
        df['created'] = pd.to_datetime(df['created'], format='%Y-%m-%d %H:%M:%S')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        df['type'] = np.where(df['retweetId'].isnull() == False, 'retweet',
                              np.where(df['replyId'].isnull() == False, "reply", "original"))


        df = df.sort_values(by=['created'])
        df = df.reset_index(drop=True)

        self.ch.save_dataframe(df, self.dm.path_dataset + filename)
        self.lm.printl(f"{file_name}. normalize_data finish.")

        return df

    def __extract_url_uk2019(self, df, filename, known_url):
        # select post with URLs
        not_nan_url = df[df['urls'].isnull() == False].copy()
        # transform url_list, which is a list in a string format in a real list with built-in function literal_eval
        not_nan_url['url_list'] = not_nan_url['urls'].apply(literal_eval)

        # Apply the function to take the url that is inside a dictionary {"expanded_url": URL}
        not_nan_url['url_list'] = not_nan_url['url_list'].apply(self.__extract_element, c='expanded_url')

        # select only original tweets
        # ou = not_nan_url.loc[(not_nan_url['replyId'].isnull() == True) & (not_nan_url['retweetId'].isnull() == True)]
        filter_df = not_nan_url[['id', 'userId', 'created', 'url_list', 'type']].copy()
        return not_nan_url
    

    def __extract_url_ira(self, df, filename, known_url):
        # select post with URLs, removing rows with NaN or empty list
        # not_nan_url = df[df['url_list'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        not_nan_url = df[
            df['url_list'].apply(
                lambda x: (
                    isinstance(x, list) and len(x) > 0      # real list, non-empty
                ) or (
                    isinstance(x, str) and x.strip() not in ['[]', '']  # non-empty string, not '[]'
                )
            )
        ]

        filter_df = not_nan_url[['id', 'userId', 'created', 'url_list', 'type', 'class']].copy()
        return filter_df

    def __extract_hashtag_uk2019(self, df, filename):
        not_nan_hashtag = df[df['hashtags'].isnull() == False]
        # original tweet
        # oh = hf.loc[(hf['replyId'].isnull() == True) & (hf['retweetId'].isnull() == True)]

        not_nan_hashtag['hashtag_list'] = not_nan_hashtag['hashtags'].apply(literal_eval)
        # Apply the function to create a new column
        not_nan_hashtag['hashtag_list'] = not_nan_hashtag['hashtag_list'].apply(self.__extract_element, c='text')
        filter_df = not_nan_hashtag[['id', 'userId', 'created', 'hashtag_list', 'type']].copy()
        return filter_df
        
    def __extract_url_IORussia(self, df, filename, known_url):
        not_nan_url = df[
            df['url_list'].apply(
                lambda x: (
                    isinstance(x, list) and len(x) > 0      # real list, non-empty
                ) or (
                    isinstance(x, str) and x.strip() not in ['[]', '']  # non-empty string, not '[]'
                )
            )
        ]

        filter_df = not_nan_url[['id', 'userId', 'created', 'url_list', 'type', 'isControl']].copy()
        filter_df['url_list'] = filter_df['url_list'].apply(self.__fix_and_parse_urls)
        return filter_df

    def __extract_hashtag_ira(self, df, filename):
        # not_nan_hashtag = df[df['hashtag_list'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        not_nan_hashtag = df[
        df['hashtag_list'].apply(
                lambda x: (
                    isinstance(x, list) and len(x) > 0      # real list, non-empty
                ) or (
                    isinstance(x, str) and x.strip() not in ['[]', '']  # non-empty string, not '[]'
                )
            )
        ]
        filter_df = not_nan_hashtag[['id', 'userId', 'created', 'hashtag_list', 'type', 'class']].copy()
        return filter_df

    def __extract_hashtag_IORussia(self, df, filename, known_url):
        not_nan_hashtag = df[
            df['hashtag_list'].apply(
                lambda x: (
                    isinstance(x, list) and len(x) > 0      # real list, non-empty
                ) or (
                    isinstance(x, str) and x.strip() not in ['[]', '']  # non-empty string, not '[]'
                )
            )
        ]

        filter_df = not_nan_hashtag[['id', 'userId', 'created', 'hashtag_list', 'type', 'isControl']].copy()
        filter_df['hashtag_list'] = filter_df['hashtag_list'].apply(self.__ensure_list)
        return filter_df

    def __extract_mention_uk2019(self, df, filename):
        not_nan_mention = df[df['mentions'].isnull() == False]
        # original tweet
        # oh = mf.loc[(mf['replyId'].isnull() == True) & (mf['retweetId'].isnull() == True)]
        del df
        not_nan_mention['mention_list'] = not_nan_mention['mentions'].apply(literal_eval)

        self.lm.printl(f"{file_name}. __extract_mentions start.")
        # Apply the function to create a new column
        not_nan_mention['mention_list'] = not_nan_mention['mention_list'].apply(self.__extract_element, c='id')

        filter_df = not_nan_mention[['id', 'userId', 'created', 'mention_list', 'type']].copy()
        return filter_df
    
    def __extract_mention_IORussia(self, df, filename, known_url):
        not_nan_mention = df[
            df['mention_list'].apply(
                lambda x: (
                    isinstance(x, list) and len(x) > 0      # real list, non-empty
                ) or (
                    isinstance(x, str) and x.strip() not in ['[]', '']  # non-empty string, not '[]'
                )
            )
        ]

        filter_df = not_nan_mention[['id', 'userId', 'created', 'mention_list', 'type', 'isControl']].copy()
        
        # Convert column to lists
        filter_df['mention_list'] = filter_df['mention_list'].apply(self.__ensure_list)
        return filter_df

    def __extract_mention_ira(self, df, filename):
        # not_nan_mention = df[df['mention_list'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        not_nan_mention = df[
            df['mention_list'].apply(
                lambda x: (
                    isinstance(x, list) and len(x) > 0      # real list, non-empty
                ) or (
                    isinstance(x, str) and x.strip() not in ['[]', '']  # non-empty string, not '[]'
                )
            )
        ]
        filter_df = not_nan_mention[['id', 'userId', 'created', 'mention_list', 'type', 'class']].copy()

        return filter_df

    def __normalize_data_user_uk2019(self, df, filename):
        df.drop(columns=['_index', '_type', '_id', '_score'], inplace=True)
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
        # convert datetime column to the best format
        df['created'] = pd.to_datetime(df['created'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

        columns = ['userId', 'name', 'screenName', 'nFollowers', 'nFriends', 'description', 'created',
                   'nLikedTweets', 'nPostedTweets', 'location',
                   'botScoreSentiment', 'botScoreEnglish', 'botScoreFriend',
                   'botScoreUniversal', 'botScoreUser', 'botScoreContent',
                   'botScoreTemporal', 'botScoreNetwork', 'isBot', 'isBotEnglish',
                   'botSkipped']
        df = df[columns]

        # bool type does not support nan. while "boolean" pandas type supports nan
        # reading the columns as "boolean", I read nan values as <NA>. So i run the following code to uniforme the nan
        # Convert all NaN to pd.NA for consistency, only if needed
        # for col in ['isBot', 'isBotEnglish', 'botSkipped']:
        #     if df[col].isna().any():
        #         df[col] = df[col].astype('object').replace({np.nan: pd.NA})

        self.ch.save_dataframe(df, self.dm.path_dataset + filename)
        return df

    def __normalize_data_user_ira(self, df, filename):
        df = df.rename(columns={'screen name': 'screenName', 'creation date': 'created', 
                            '# min followers': 'nMinFollowers', '# max followers': 'nMaxFollowers', 
                            '# min friends': 'nMinFriends', '# max friends': 'nMaxFriends'})
        return df


    # Ensure all mention and hashtag list entries are actual Python lists
    def __ensure_list(self, x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)  # safely evaluate string like "['url1', 'url2']"
                if isinstance(val, list):
                    return val
                else:
                    return []
            except (ValueError, SyntaxError):
                return []
        return []


    def __fix_and_parse_urls(self, x):
        """
        Cleans and extracts URLs from messy inputs.
        Handles:
        - Real Python lists
        - String representations of lists
        - Concatenated URLs glued together without commas
        """
        urls = []

        # Case 1: already a list
        if isinstance(x, list):
            urls = x

        # Case 2: string that may represent a list
        elif isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    urls = val
                else:
                    # sometimes it's a string like "https://a.comhttps://b.com"
                    urls = [x]
            except (ValueError, SyntaxError):
                # fallback: treat it as a single string
                urls = [x]
        else:
            return []

        fixed_urls = []
        for url in urls:
            if not isinstance(url, str):
                continue

            # Split if concatenated URLs appear
            parts = re.split(r'(?=https?://)', url)
            for p in parts:
                p = p.strip()
                if p.startswith("http"):
                    fixed_urls.append(p)

        return fixed_urls


    # PUBLIC METHODS
    # ------------------------------------------------------------------------------------------------------------------
    def from_json_to_dataframe(self, filename):
        """
            Convert a JSON file in a pandas dataframe, saving as csv file. This is done in two steps: first the json file is converted
            in several csv files, each one including 100000 rows. Then these csv file are merged in a unique dataframe and saved.
            :param filename: [str] filename.json to convert.
            :return: [DataFrame] Return a pandas DataFrame of the json data.
        """
        # get the filename without the extension (json), concatenating the csv extension
        csv_filename = filename.split(".")[0] + ".csv"

        chunk_size = 1000000
        i = 0
        data = []
        df = pd.DataFrame()

        # STEP 1: Convert json in many csv files
        self.lm.printl(f"{file_name}. from_json_to_dataframe Start converting JSON file: {self.dm.path_dataset}{filename}")
        with open(self.dm.path_dataset + filename) as f:
            for line in f:
                data.append(json.loads(line))
                i += 1
                if i % chunk_size == 0:
                    self.lm.printl(f"{file_name}. Pandas dataframe number: {str(i / chunk_size)}")
                    # json_normalize explodes the object in multiple columns
                    chunk_df = pd.json_normalize(data)
                    data = []
                    self.ch.save_dataframe(chunk_df, self.dm.path_temp + str(i) + "_" + csv_filename)

            # in the last chunk if there are less than chunk_size rows, it does not save, so i have to save these rows
            chunk_df = pd.json_normalize(data)
            data = []
            self.ch.save_dataframe(chunk_df, self.dm.path_temp + str(i) + "_" + csv_filename)
        self.lm.printl(f"{file_name}. Finish writing all csv files.")

        # STEP 2: merge all csv files in a unique dataframe
        self.lm.printl(f"{file_name}. Starting merging all csv files.")
        csv_files = [pos_csv for pos_csv in os.listdir(self.dm.path_temp) if pos_csv.endswith('.csv')]
        for cf in csv_files:
            temp_df = pd.read_csv(self.dm.path_temp + cf)
            df = pd.concat([df, temp_df], ignore_index=True)

            self.ch.save_dataframe(df, self.dm.path_dataset + "final_" + csv_filename)
        self.lm.printl(
            f"{file_name}. Finish merging all csv files in a unique file:  {self.dm.path_dataset}final_{csv_filename}")
        return df

    def download_ES_data(self, elastic_info):
        """
            Read data from Elasticsearch or from a CSV.
            :param filename: [str] Filename of a csv file.
            :param elastic_info: [dict] Elasticsearch info, including the following field.
            username_index: [str] Username of the credentials to access the ES index.
            index_name: [str] Name of the index to be accessed.
            start_date: ['str'] Date of the start period from which retrieving data from Elasticsearch. Format: '%Y-%m-%d %H:%M:%S'.
            end_date: ['str'] Date of the end period from which retrieving data from Elasticsearch. Format: '%Y-%m-%d %H:%M:%S'
            :return: [DataFrame] Return a pandas DataFrame of the json data. An empty dataset is returned in case no data are retrieved from Elasticsearch or in case of
            specific methods, which write data directly on a file.
        """

        self.type_query = elastic_info['type_query']
        self.username_index = elastic_info['username_index']
        self.index_name = elastic_info['index_name']
        self.es = ESManager(self.username_index)
        self.ind = ESIndexManager(self.es, self.index_name)
        self.lm.printl(self.es.existIndex(self.index_name))
        self.lm.printl(self.ind.getCount())

        df = pd.DataFrame()

        if self.type_query == 'all':
            df = self.__searchElasticAll()
        elif self.type_query == 'date':
            startDate = elastic_info['start_date']
            endDate = elastic_info['end_date']

            df = self.__searchElasticByDate(startDate, endDate)
        elif self.type_query == 'dateSources':
            startDate = elastic_info['startDate']
            endDate = elastic_info['endDate']
            sourceList = elastic_info['sourceList']
            filename = elastic_info['filename']

            # write data on a json file, exploiting a generator
            self.__searchElasticSourcesByDate(startDate, endDate, sourceList, filename)
        elif self.type_query == "dateUsersSources":
            startDate = elastic_info['startDate']
            endDate = elastic_info['endDate']
            sourceList = elastic_info['sourceList']
            userList = elastic_info['userList']
            filename = elastic_info['filename']

            # write data on a json file, exploiting a generator
            self.__searchElasticSourcesUsersByDate(startDate, endDate, sourceList, userList, filename)
        elif self.type_query == 'allSources':
            sourceList = elastic_info['sourceList']
            filename = elastic_info['filename']

            self.__searchElasticSources(sourceList, filename)

        self.lm.printl(f"{file_name}. Elasticsearch data retrieved.")
        return df

    def normalize_data(self, df, filename):
        """
            Normalize data, performing some changes in the dataframe, e.g., renaming columns or changing columns' type.
            :param df: [DataFrame] DataFrame on which performing operations.
            :param filename: [str] Filename of a csv file.
            :return: [DataFrame] Return a pandas DataFrame with the performed changes
        """
        self.lm.printl("InputManager-normalize_data start.")
        if self.dataset_name == "uk2019":
            df = self.__normalize_data_uk2019(df, filename)
            return df
        elif self.dataset_name == "ira":
            df = self.__normalize_data_ira(df, filename)
            return df
        elif self.dataset_name == "IORussia":
            df = self.__normalize_data_IORussia(df, filename)
            return df
        elif self.dataset_name == "moltbook":
            df = self.__normalize_data_moltbook(df, filename)
            return df
    

    def normalize_data_text(self, df, filename):
        """
            Normalize data text, performing some changes in the dataframe, e.g., renaming columns or changing columns' type.
            :param df: [DataFrame] DataFrame on which performing operations.
            :param filename: [str] Filename of a csv file.
            :return: [DataFrame] Return a pandas DataFrame with the performed changes
            :return: [DataFrame] Return a pandas DataFrame with the performed changes
        """
        self.lm.printl(f"{file_name}: normalize_data_text start.")
        rename_dict = {
            "id_str": "id",
            "created_at": "created",
            "user.id_str": "userId",
        }
        df = df.rename(columns=rename_dict)
        df = df[df['created'].isnull()==False]
        df = df[df['created']!='0']

        # convert datetime column to the best format
        df['created'] = pd.to_datetime(df['created'], format='%a %b %d %H:%M:%S +0000 %Y')
        df['created'] = df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['type'] = np.where(df['retweetId'].isnull() == False, 'retweet', np.where(df['replyId'].isnull() == False, "reply", "original"))
        df = df.sort_values(by=['created'])
        df = df[['id', 'userId', 'created', 'text', 'retweet_count', 'favorite_count', 'reply_count', 'quote_count', 'type']]
        self.ch.save_dataframe(df, self.dm.path_dataset + filename)
        self.lm.printl(f"{file_name}. normalize_data_text finish.")

   

    def normalize_data_user(self, df, filename):
        """
            Normalize data of the users, performing some changes in the dataframe, e.g., renaming columns or changing columns' type.
            :param df: [DataFrame] DataFrame on which performing operations.
            :param filename: [str] Filename of a csv file.
            :return: [DataFrame] Return a pandas DataFrame with the performed changes
            :return: [DataFrame] Return a pandas DataFrame with the performed changes
        """
        self.lm.printl(f"{file_name}: normalize_data_user start.")

        if self.dataset_name == "uk2019":
            df = self.__normalize_data_user_uk2019(df, filename)
        elif self.dataset_name == "ira":
            df =  self.__normalize_data_user_ira(df, filename)
        
        self.lm.printl(f"{file_name}. normalize_data_user completed.")
        return df

    def extract_url_dataset(self, df, filename, known_url, parse_urls=True):
        
        """
            Given a dataframe of posts published by several users, it extracts URLs of original tweets (retweet and reply excluded)
             contained in each post.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :param excludeDomainList: [list, optional] List of domainUrls to be excluded from the dataframe
            :return: [DataFrame] Return a pandas DataFrame with extracted URLs.
        """
        self.lm.printl(f"{file_name}. extract_url_dataset start.")
        if self.dataset_name == "uk2019":
            filter_df = self.__extract_url_uk2019(df, filename, known_url)
        elif self.dataset_name == "ira":
            filter_df =  self.__extract_url_ira(df, filename, known_url)
        elif self.dataset_name == "IORussia":
            filter_df =  self.__extract_url_IORussia(df, filename, known_url)
    
        # each tweet can contain more than one URL, which is included in the url_list column, which is a list. I explode
        # this column, creating a row for each URL
        result_df = filter_df.explode('url_list')
        result_df = result_df.rename(columns={"url_list": "url"})
        
        # Drop empty URLs (None, empty string, NaN)
        result_df = result_df[result_df['url'].notna() & (result_df['url'] != '')]
        self.lm.printl(result_df.shape)
        self.lm.printl(result_df.columns)
        if parse_urls:
            # i work on the unique url df, in this way i avoid to process the same url multiple time,
            # it reduces 50% the number of urls to be processed
            # but it is needed a final merge, to attach the results to the original dataframe
            url_df = pd.DataFrame(result_df['url'].unique(), columns=["url"])
            self.lm.printl(f"{file_name}. Number of urls to be processed: {str(url_df.shape[0])}")
            # 1) I parse the url. most of the url are known domain and must not be unshortened
            start_time = time.time()
            url_df['domainUrl'], url_df['pathUrl'] = zip(*url_df['url'].swifter.apply(self.__parse_url))
            self.lm.printl(f"{file_name}. First parsing url completed, seconds {str((time.time() - start_time))}")

            # i separate in two dataframes the known and unknown url domains
            # the known ones must not be unshortened
            known_df = url_df[url_df['domainUrl'].isin(known_url)]
            not_known_df = url_df[~url_df['domainUrl'].isin(known_url)]

            known_df['resolved_url'] = known_df['url']

            self.lm.printl(f"{file_name}. Unshortening start")
            # Unshorten unknown urls
            start_time = time.time()
            not_known_df['resolved_url'] = not_known_df['url'].swifter.apply(self.__unshorten_url)
            self.lm.printl(f"{file_name}. Unshortening completed, seconds {str((time.time() - start_time))}")

            # redo the parsing on the resolved url. I replace the previous parsing
            start_time = time.time()
            not_known_df['domainUrl'], not_known_df['pathUrl'] = zip(*not_known_df['resolved_url'].swifter.apply(self.__parse_url))
            self.lm.printl(f"{file_name}. Second parsing url completed, seconds {str((time.time() - start_time))}")

            # reconstruct the dataframe with both parts
            url_df = pd.concat([known_df, not_known_df]).reset_index()

            # I perform the inner join of the unique url dataframe with the original dataframe
            self.lm.printl(f"{file_name}. Start merging url with original dataframe.")
            result_df = pd.merge(result_df, url_df, on="url")

            result_df = result_df.dropna()
            result_df = result_df.drop(columns=['index'])
        else:
            result_df = result_df.rename(columns={"url": "domainUrl"})

        self.ch.save_dataframe(result_df, self.dm.path_dataset + filename)
        self.lm.printl(f"Number of extracted URLs: {len(result_df)}, unique userIds: {len(result_df['userId'].unique())}")
        self.lm.printl(f"{file_name}. extract_url_dataset completed.")

        
    def extract_text_dataset(self, df, filename):
        df = df[df['text'].isnull() == False]
        em = EmbeddingManager()
        em.build_multilingual_embeddings(df, text_col="content", output_path=f"{self.dm.path_dataset}{filename}",model_name="intfloat/multilingual-e5-large", batch_size=32)

    def filter_content_df(self, df, c, excludeList, filename):
        self.lm.printl(f"{file_name}. filter_content_df start.")
        self.lm.printl(f"{file_name}. Before filtering #rows: {str(df.shape[0])}, # distinct{c}s: {str(len(df[c].unique()))}")

        # Convert each string in the list to lowercase
        excludeList = [s.lower() for s in excludeList]

        # if I want to exclude specific domains/hashatgs/mentions
        df = df[~df[c].isin(excludeList)]

        self.ch.save_dataframe(df, self.dm.path_dataset + filename)

        self.lm.printl(f"{file_name}. After filtering #rows: {str(df.shape[0])}, # distinct{c}s: {str(len(df[c].unique()))}")
        self.lm.printl(f"{file_name}. filter_content_df completed.")
        return df


    def extract_hashtag_dataset(self, df, filename):
        """
            Given a dataframe of posts published by several users, it extracts hashtags of original tweets (retweet and reply excluded)
             contained in each post.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :return: [DataFrame] Return a pandas DataFrame with extracted hashtags.
        """
        self.lm.printl(f"{file_name}. extract_hashtag_dataset start.")
        if self.dataset_name == "uk2019":
            filter_df = self.__extract_hashtag_uk2019(df, filename)
        elif self.dataset_name == "ira":
            filter_df = self.__extract_hashtag_ira(df, filename)
        elif self.dataset_name == "IORussia":
            filter_df = self.__extract_hashtag_IORussia(df, filename, known_url=None)
        
        result_df = filter_df.explode('hashtag_list')
        result_df = result_df.rename(columns={"hashtag_list": "hashtag"})
        # result_df = result_df.drop(columns=['hashtags'])

        # Drop empty URLs (None, empty string, NaN)
        result_df = result_df[result_df['hashtag'].notna() & (result_df['hashtag'] != '')]

        result_df['hashtag'] = result_df['hashtag'].str.strip().str.lower()
        result_df = result_df.dropna()
        self.ch.save_dataframe(result_df, self.dm.path_dataset + filename)
        self.lm.printl(
            f"Number of extracted hashtags: {len(result_df)}, unique userIds: {len(result_df['userId'].unique())}")
        self.lm.printl(f"{file_name}. extract_hashtag_dataset finish.")
        return result_df


    def extract_mention_dataset(self, df, filename):
        """
            Given a dataframe of posts published by several users, it extracts mentions of original tweets (retweet and reply excluded)
             contained in each post.
            :param df: [DataFrame] DataFrame of posts.
            :param filename: [str] Csv file to save the pandas dataframe
            :return: [DataFrame] Return a pandas DataFrame with extracted mentions.
        """
        self.lm.printl(f"{file_name}. extract_mention_dataset start.")
        if self.dataset_name == "uk2019":
            filter_df = self.__extract_mention_uk2019(df, filename)
        elif self.dataset_name == "ira":
            filter_df = self.__extract_mention_ira(df, filename)
        elif self.dataset_name == "IORussia":
            filter_df = self.__extract_mention_IORussia(df, filename, known_url=None)
        
        
        self.lm.printl(f"{file_name}. explode mention_list.")
        result_df = filter_df.explode('mention_list')
        result_df = result_df.rename(columns={"mention_list": "mention"})
        # Drop empty URLs (None, empty string, NaN)
        result_df = result_df[result_df['mention'].notna() & (result_df['mention'] != '')]
        # result_df = result_df.drop(columns=['mentions'])

        result_df = result_df.dropna()
        self.ch.save_dataframe(result_df, self.dm.path_dataset + filename)


        self.lm.printl(f"Number of extracted mentions: {len(result_df)}, unique userIds: {len(result_df['userId'].unique())}")
        self.lm.printl(f"{file_name}. extract_mention_dataset finish.")
        return result_df

    def extract_retweet_dataset(self, df, filename):
        self.lm.printl(f"{file_name}. extract_retweet_dataset start.")
        filter_df = self.__extract_retweet_reply(df, filename, 'retweetId', 'retweet')
        self.lm.printl(f"{file_name}. extract_retweet_dataset completed.")
        return filter_df

    def extract_reply_dataset(self, df, filename):
        self.lm.printl(f"{file_name}. extract_reply_dataset start.")
        filter_df =  self.__extract_retweet_reply(df, filename, 'replyId', 'reply')
        self.lm.printl(f"{file_name}. extract_reply_dataset completed.")
        return filter_df