from __future__ import annotations

from Objects.CoAction.CoAction import CoAction
from Objects.Filter.Filter import Filter
from Objects.TimeWindow.TimeWindow import TimeWindow
from configs.base import PipelineConfig


def get_config() -> PipelineConfig:
    """
    Build the Iran5 Twitter configuration used by the external main script.

    :return: [PipelineConfig] Iran5 pipeline configuration.
    """
    dataset_name = "iran5"

    tw = TimeWindow(
        type_output_network="merged",
        type_time_window="OTW",
        tw_str="7d",
        tw_slide_interval_str="6d",
        type_merge="average",
    )

    list_ca = [
        CoAction("co-url-domain", "tfidf_cosine_similarity"),
        CoAction("co-mention", "tfidf_cosine_similarity"),
        CoAction("co-hashtag", "tfidf_cosine_similarity"),
        CoAction("co-reply", "tfidf_cosine_similarity"),
        CoAction("co-retweet", "tfidf_cosine_similarity"),
    ]

    no_filter = {
        "co-url-domain": None,
        "co-mention": None,
        "co-hashtag": None,
        "co-reply": None,
        "co-retweet": None,
    }

    n_action_filter = {
        "co-url-domain": Filter("merge_filter_action", 2, None),
        "co-mention": Filter("merge_filter_action", 2, None),
        "co-hashtag": Filter("merge_filter_action", 2, None),
        "co-reply": Filter("merge_filter_action", 2, None),
        "co-retweet": Filter("merge_filter_action", 2, None),
    }

    final_filter = {
        "co-url-domain": Filter("median", None, Filter("merge_filter_action", 2, None)),
        "co-mention": Filter("median", None, Filter("merge_filter_action", 2, None)),
        "co-hashtag": Filter("median", None, Filter("merge_filter_action", 2, None)),
        "co-reply": Filter("median", None, Filter("merge_filter_action", 2, None)),
        "co-retweet": Filter("median", None, Filter("merge_filter_action", 2, None)),
    }

    return PipelineConfig(
        dataset_name=dataset_name,
        known_url=[],
        exclude_domain_list=[],
        exclude_hashtag_list=[],
        exclude_mention_dict={},
        filter_dataset={
            "co-url-domain": False,
            "co-mention": False,
            "co-hashtag": False,
            "co-reply": False,
            "co-retweet": False,
        },
        user_fraction=0.05,
        type_filter="top_co_action_merge",
        user_selection_fractions=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        tw=tw,
        co_action_list=["co-url-domain", "co-mention", "co-hashtag", "co-reply", "co-retweet"],
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
            # "infomap": [None],
        },
        multiplex_algorithm_dict={
            "glouvain": [{"omega": 0.1, "gamma": 1}],
        },
        similarity_parallelize_window=70,
    )


CONFIG = get_config()
