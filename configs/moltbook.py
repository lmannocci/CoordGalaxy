from __future__ import annotations

from Objects.CoAction.CoAction import CoAction
from Objects.Filter.Filter import Filter
from Objects.TimeWindow.TimeWindow import TimeWindow
from configs.base import PipelineConfig


def get_config() -> PipelineConfig:
    """
    Build the Moltbook configuration used by the external main script.

    :return: [PipelineConfig] Moltbook pipeline configuration.
    """
    dataset_name = "moltbook"

    tw = TimeWindow(
        type_output_network="merged",
        type_time_window="OTW",
        tw_str="1d",
        tw_slide_interval_str="12h",
        type_merge="average",
    )

    list_ca = [
        CoAction("co-comment", "tfidf_cosine_similarity"),
        CoAction("co-commentText", "average_cosine_similarity"),
        CoAction("co-commentURL", "tfidf_cosine_similarity"),
        CoAction("co-postText", "average_cosine_similarity"),
        CoAction("co-postURL", "tfidf_cosine_similarity"),
    ]

    no_filter = {
        "co-comment": None,
        "co-commentText": None,
        "co-commentURL": None,
        "co-postText": None,
        "co-postURL": None,
    }

    n_action_filter = {
        "co-comment": Filter("merge_filter_action", 2, None),
        "co-commentText": Filter("merge_filter_action", 2, None),
        "co-commentURL": Filter("merge_filter_action", 2, None),
        "co-postText": Filter("merge_filter_action", 12, None),
        "co-postURL": Filter("merge_filter_action", 5, None),
    }

    final_filter = {
        "co-comment": Filter("th", 0.06, Filter("merge_filter_action", 2, None)),
        "co-commentText": Filter("th", 0.85, Filter("merge_filter_action", 2, None)),
        "co-commentURL": Filter("th", 0.32, Filter("merge_filter_action", 2, None)),
        "co-postText": Filter("th", 0.87, Filter("merge_filter_action", 12, None)),
        "co-postURL": Filter("th", 0.99, Filter("merge_filter_action", 5, None)),
    }

    return PipelineConfig(
        dataset_name=dataset_name,
        known_url=[],
        exclude_domain_list=[],
        exclude_hashtag_list=[],
        exclude_mention_dict={},
        filter_dataset={
            "co-comment": False,
            "co-commentText": False,
            "co-commentURL": False,
            "co-postText": False,
            "co-postURL": False,
        },
        user_fraction=None,
        type_filter="top_co_action_merge",
        user_selection_fractions=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        tw=tw,
        co_action_list=["co-comment", "co-commentText", "co-commentURL", "co-postText", "co-postURL"],
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
            # "ginfomap": [{"interlayer_weight": 0.15}],

            # "flat_ec_infomap": [None],
            # "flat_nw_infomap": [None],
            # "flat_weighted_sum_infomap": [None],
            # "flat_and_weighted_sum_infomap": [None],

            "glouvain": [{"omega": 0.1, "gamma": 1}],

            # "flat_ec_louvain": [{"resolution": 1}],
            # "flat_nw_louvain": [{"resolution": 1}],
            # "flat_weighted_sum_louvain": [{"resolution": 1}],
            # "flat_and_weighted_sum_louvain": [{"resolution": 1}],
        },
        text_similarity_threshold=0.7,
        text_similarity_chunk_size=5000,
        similarity_parallelize_window=70,
    )


CONFIG = get_config()
