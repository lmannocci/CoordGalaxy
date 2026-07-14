from __future__ import annotations

from datetime import datetime


def select_sources_es() -> list[str]:
    """
    Return tweet source fields used by legacy Elasticsearch exports.

    :return: [list[str]] Elasticsearch source field names.
    """
    tweet_keys = ["created_at", "id_str", "user.id_str"]

    hashtag_keys = [
        "entities.hashtags.text",
        "extended_tweet.entities.hashtags.text",  # .text
        "retweeted_status.entities.hashtags.text",  # .text
        "retweeted_status.extended_tweet.entities.hashtags.text",  # .text
    ]

    mention_keys = [
        "entities.user_mentions.id",  # .id
        "extended_tweet.entities.user_mentions.id",  # .id
        "retweeted_status.entities.user_mentions.id",  # .id
        "retweeted_status.extended_tweet.entities.user_mentions.id",  # .id
    ]

    url_keys = [
        "entities.urls.expanded_url",  # .expanded_url
        "extended_tweet.entities.urls.expanded_url",  # .expanded_url
        "retweeted_status.entities.urls.expanded_url",  # .expanded_url
        "retweeted_status.extended_tweet.entities.urls.expanded_url",  # expanded_url
    ]

    retweet_keys = ["retweeted_status.id_str", "retweeted_status.created_at", "retweeted_status.user.id_str"]
    reply_keys = ["in_reply_to_status_id_str", "in_reply_to_user_id_str"]

    return tweet_keys + retweet_keys + reply_keys + hashtag_keys + mention_keys + url_keys


def select_sources_info_tweet_es() -> list[str]:
    """
    Return source fields for tweet metadata exports.

    :return: [list[str]] Elasticsearch tweet metadata source field names.
    """
    return [
        "id_str",
        "user.id_str",
        "text",
        "favorite_count",
        "quote_count",
        "reply_count",
        "retweet_count",
        "created_at",
    ]


def select_sources_info_user_es() -> list[str]:
    """
    Return source fields for user metadata exports.

    :return: [list[str]] Elasticsearch user metadata source field names.
    """
    return [
        "id_str",
        "name",
        "screen_name",
        "description",
        "location",
        "favourites_count",
        "followers_count",
        "friends_count",
        "statuses_count",
        "botometer",
        "created_at",
    ]


def get_formatted_date(date: str) -> str:
    """
    Convert a YYYY-mm-dd HH:MM:SS date into the legacy Twitter/Elasticsearch date string.

    :param date: [str] Date in YYYY-mm-dd HH:MM:SS format.
    :return: [str] Formatted date string, for example Tue Nov 12 00:00:00 +0000 2019.
    """
    input_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return input_date.strftime("%a %b %d %H:%M:%S +0000 %Y")


# Backward-compatible names used by older config files.
select_sources_ES = select_sources_es
select_sources_info_tweet_ES = select_sources_info_tweet_es
select_sources_info_user_ES = select_sources_info_user_es
