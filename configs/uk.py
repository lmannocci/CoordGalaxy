from __future__ import annotations

from Objects.CoAction.CoAction import CoAction
from Objects.Filter.Filter import Filter
from Objects.TimeWindow.TimeWindow import TimeWindow
from configs.base import PipelineConfig
from utils.ElasticSearch.source_fields import select_sources_info_user_ES


def get_config() -> PipelineConfig:
    """
    Build the UK election configuration used by the external main script.

    :return: [PipelineConfig] UK pipeline configuration.
    """
    dataset_name = "uk"

    tw = TimeWindow(
        type_output_network="merged",
        type_time_window="OTW",
        tw_str="6h",
        tw_slide_interval_str="5h",
        type_merge="average",
    )

    list_ca = [
        CoAction("co-retweet", "tfidf_cosine_similarity"),
        CoAction("co-reply", "tfidf_cosine_similarity"),
        CoAction("co-url-domain", "tfidf_cosine_similarity"),
        CoAction("co-mention", "tfidf_cosine_similarity"),
        CoAction("co-hashtag", "tfidf_cosine_similarity"),
    ]

    no_filter = {
        "co-retweet": None,
        "co-reply": None,
        "co-url-domain": None,
        "co-mention": None,
        "co-hashtag": None,
    }

    n_action_filter = {
        "co-retweet": Filter("merge_filter_action", 26, None),
        "co-reply": Filter("merge_filter_action", 4, None),
        "co-url-domain": Filter("merge_filter_action", 3, None),
        "co-mention": Filter("merge_filter_action", 48, None),
        "co-hashtag": Filter("merge_filter_action", 14, None),
    }

    final_filter = {
        "co-retweet": Filter("median", None, Filter("merge_filter_action", 26, None)),
        "co-reply": Filter("median", None, Filter("merge_filter_action", 4, None)),
        "co-url-domain": Filter("median", None, Filter("merge_filter_action", 3, None)),
        "co-mention": Filter("median", None, Filter("merge_filter_action", 48, None)),
        "co-hashtag": Filter("median", None, Filter("merge_filter_action", 14, None)),
    }

    known_url = [
        "twitter.com",
        "cards.twitter.com",
        "youtube.com",
        "instagram.com",
        "facebook.com",
        "open.spotify.com",
        "google.com",
        "reddit.com",
        "play.google.com",
        "bing.com",
        "google.co.uk",
        "paper.li",
        "theguardian.com",
        "bbc.co.uk",
        "independent.co.uk",
        "mirror.co.uk",
        "telegraph.co.uk",
        "thesun.co.uk",
        "mobile.twitter.com",
        "metro.co.uk",
        "dailymail.co.uk",
        "express.co.uk",
        "thetimes.co.uk",
        "bbc.com",
        "news.sky.com",
    ]

    return PipelineConfig(
        dataset_name=dataset_name,
        known_url=known_url,
        exclude_domain_list=[
            "twitter.com",
            "cards.twitter.com",
            "youtube.com",
            "instagram.com",
            "facebook.com",
            "open.spotify.com",
            "google.com",
            "reddit.com",
            "play.google.com",
            "bing.com",
            "google.co.uk",
        ],
        exclude_hashtag_list=[
            "GE2019",
            "GeneralElection19",
            "GeneralElection19",
            "GeneralElection2019",
            "VoteLabour",
            "VoteLabour2019",
            "ForTheMany",
            "ForTheManyNotTheFew",
            "ChangeIsComing",
            "RealChange",
            "VoteConservative",
            "VoteConservative2019",
            "BackBoris",
            "GetBrexitDone",
        ],
        exclude_mention_dict={
            "jeremycorbyn": "117777690",
            "UKLabour": "14291684",
            "3131144855": "BorisJohnson",
            "Conservatives": "14281853",
        },
        filter_dataset={
            "co-retweet": False,
            "co-reply": False,
            "co-url-domain": True,
            "co-mention": True,
            "co-hashtag": True,
        },
        user_fraction=0.05,
        type_filter="top_co_action_merge",
        user_selection_fractions=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        tw=tw,
        co_action_list=["co-retweet", "co-reply", "co-mention", "co-hashtag", "co-url-domain"],
        similarity_function="tfidf_cosine_similarity",
        list_ca=list_ca,
        co_action_filters={
            "no_filter": no_filter,
            "n_action": n_action_filter,
            "final": final_filter,
        },
        metrics_to_compute=[
            "connected_components",
            "weight_statistics",
            "nNodes",
            "nEdges",
            "directed",
            "sizeLargestComponent",
            "density",
            "clusteringCoefficient",
            "averagePathLength",
            "diameter",
        ],
        metrics_node_to_compute=[
            "degree_centrality",
            "eigenvector_centrality",
            "page_rank",
            "local_clustering_coefficient",
        ],
        single_layer_algorithm_dict={
            "louvain": [{"resolution": 1}],
            "infomap": [None],
        },
        multiplex_algorithm_dict={
            "ginfomap": [{"interlayer_weight": 0.15}],
            "flat_ec_infomap": [None],
            "flat_nw_infomap": [None],
            "flat_weighted_sum_infomap": [None],
            "flat_and_weighted_sum_infomap": [None],
            "glouvain": [{"omega": 0.1, "gamma": 1}],
            "flat_ec_louvain": [{"resolution": 1}],
            "flat_nw_louvain": [{"resolution": 1}],
            "flat_weighted_sum_louvain": [{"resolution": 1}],
            "flat_and_weighted_sum_louvain": [{"resolution": 1}],
        },
        similarity_parallelize_window=70,
        elastic_info={
            "username_index": "nizzoli",
            "index_name": "ukelections19_users",
            "type_query": "allSources",
            "sourceList": select_sources_info_user_ES(),
            "filename": "1_uk_final_users.csv",
        },
    )


CONFIG = get_config()
