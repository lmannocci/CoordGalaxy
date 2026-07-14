from __future__ import annotations

from typing import Any

from CharacterizationManager.CharacterizationManager import CharacterizationManager
from CommunityDetectionManager.CommunityDetectionManager import CommunityDetectionManager
from FilterGraphManager.FilterGraphManager import FilterGraphManager
from InputManager import InputManager
from NetworkManager import NetworkManager
from Objects.CDAlgorithm.CDAlgorithm import CDAlgorithm
from SelectionUserManager import SelectionUserManager
from SimilarityFunctionManager import SimilarityFunctionManager
from Objects.CoAction.CoAction import CoAction
from configs import PipelineConfig, load_config
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.common_variables import dtype
from utils.pipeline_io import DatasetPaths, build_paths, read_original_file, read_temp_file


SELECTED_DATASET = "iran5"
RAW_TWEETS_FILE = "iran5.csv"
NORMALIZED_TWEETS_FILE = "normalized_tweets.csv"


def preprocess_iran5_input(config: PipelineConfig, paths: DatasetPaths, ch: Checkpoint, im: InputManager) -> None:
    """
    Normalize Iran5 source tweets and extract the requested Twitter co-action artifacts.

    :param config: Iran5 pipeline configuration.
    :param paths: Iran5 path bundle.
    :param ch: Checkpoint instance.
    :param im: InputManager configured for the Iran5 dataset.
    :return: None. Normalized CSVs are saved in temp_data and co-action CSV files in co_action_data.
    """
    df = read_original_file(ch, paths, RAW_TWEETS_FILE)
    im.normalize_data(df, filename=NORMALIZED_TWEETS_FILE)

    normalized_df = read_temp_file(ch, paths, NORMALIZED_TWEETS_FILE)
    im.extract_url_dataset(normalized_df, "URL.csv", config.known_url, parse_urls=False)
    im.extract_mention_dataset(normalized_df, "mention.csv")
    im.extract_hashtag_dataset(normalized_df, "hashtag.csv")
    im.extract_reply_dataset(normalized_df, "reply.csv")
    im.extract_retweet_dataset(normalized_df, "retweet.csv")


def run_user_selection(config: PipelineConfig, lm: LogManager) -> None:
    """
    Optionally analyze and apply user selection for Iran5.

    :param config: Iran5 pipeline configuration.
    :param lm: Log manager.
    :return: None. Selection artifacts are saved by SelectionUserManager when enabled.
    """
    for selection_fraction in config.user_selection_fractions:
        su = SelectionUserManager(config.dataset_name, selection_fraction, config.type_filter, config.co_action_list)
        su.analyze_user_selection(config.filter_dataset)

    if config.user_fraction is None:
        lm.printl("main_iran5. user_fraction=None, skipping user filtering and using all users.")
        return

    su = SelectionUserManager(config.dataset_name, config.user_fraction, config.type_filter, config.co_action_list)
    su.plot_overlapping_percentage_users()
    su.plot_number_users()
    su.apply_user_selection(config.filter_dataset)


def compute_similarity_edges(config: PipelineConfig) -> None:
    """
    Compute similarity edge lists for configured Iran5 co-actions.

    :param config: Iran5 pipeline configuration.
    :return: None. Similarity outputs are saved by SimilarityFunctionManager.
    """
    ca_by_name = {ca.get_co_action(): ca for ca in config.list_ca}
    for co_action in config.co_action_list:
        sm = SimilarityFunctionManager(
            config.dataset_name,
            config.user_fraction,
            config.type_filter,
            config.tw,
            ca_by_name.get(co_action, CoAction(co_action, config.similarity_function)),
            parallelize_window=config.similarity_parallelize_window,
            text_similarity_threshold=config.text_similarity_threshold,
            text_similarity_chunk_size=config.text_similarity_chunk_size,
        )
        sm.compute_similarity()


def characterize_unfiltered_networks(config: PipelineConfig) -> None:
    """
    Characterize unfiltered Iran5 similarity networks.

    :param config: Iran5 pipeline configuration.
    :return: None. Characterization outputs are saved by CharacterizationManager.
    """
    chm = CharacterizationManager(
        config.dataset_name,
        config.user_fraction,
        config.type_filter,
        config.tw,
        config.list_ca,
        config.co_action_filters["no_filter"],
    )
    chm.compute_threshold_statistics(2, 480, 5, "nAction")
    chm.plot_threshold_statistics("nAction", 10)
    chm.select_threshold_statistics(0.04, 0.3, 0.02, False, "nAction", "node")
    chm.select_threshold_statistics(10000, 20000, 1000, True, "nAction", "node")
    chm.compute_network_metrics(config.metrics_to_compute)


def filter_by_action_count_and_analyze_weight_thresholds(config: PipelineConfig) -> None:
    """
    Filter Iran5 networks by action count and analyze weight thresholds.

    :param config: Iran5 pipeline configuration.
    :return: None. Filtered graphs and weight-threshold characterization outputs are saved.
    """
    action_filter = config.co_action_filters["n_action"]
    fm = FilterGraphManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, action_filter)
    fm.filter_graph()

    chm = CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, action_filter)
    chm.compute_threshold_statistics(0.01, 0.99, 0.01, "w_")
    chm.plot_threshold_statistics("w_", 0.01)
    chm.select_threshold_statistics(10000, 100000, 10000, True, "w_", "edge")
    chm.compute_network_metrics(config.metrics_to_compute)


def apply_final_network_filters(config: PipelineConfig) -> None:
    """
    Apply the final Iran5 co-action filters.

    :param config: Iran5 pipeline configuration.
    :return: None. Filtered edge lists are saved by FilterGraphManager.
    """
    final_filter = config.co_action_filters["final"]
    fm = FilterGraphManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    fm.filter_graph()


def characterize_final_filtered_networks(config: PipelineConfig) -> None:
    """
    Compute network metrics for the final filtered Iran5 co-action networks.

    :param config: Iran5 pipeline configuration.
    :return: None. Network metric tables are saved by CharacterizationManager.
    """
    final_filter = config.co_action_filters["final"]
    chm = CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    chm.compute_network_metrics(config.metrics_to_compute)


def create_final_network_artifacts(config: PipelineConfig) -> None:
    """
    Create weighted graph, multiplex graph, and Gephi artifacts for Iran5.

    :param config: Iran5 pipeline configuration.
    :return: None. Network artifacts are saved by NetworkManager.
    """
    final_filter = config.co_action_filters["final"]
    nm = NetworkManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    nm.create_weighted_graph()
    nm.create_weighted_multiplex_network()
    nm.save_gephi_network()


def compare_final_network_layers(config: PipelineConfig) -> None:
    """
    Compute and plot multiplex layer comparison for the final Iran5 network.

    :param config: Iran5 pipeline configuration.
    :return: None. Layer-comparison outputs are saved by CharacterizationManager.
    """
    final_filter = config.co_action_filters["final"]
    chm = CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    chm.get_ML_layer_comparison()


def build_cda(algorithm: str, parameters: dict[str, Any] | None) -> CDAlgorithm:
    """
    Build a community-detection algorithm object from configured parameters.

    :param algorithm: Algorithm name.
    :param parameters: Parameter dictionary, or None for algorithms without parameters.
    :return: CDAlgorithm instance.
    """
    if parameters is None:
        return CDAlgorithm(algorithm)
    return CDAlgorithm(algorithm, parameters)


def run_single_layer_community_detection(config: PipelineConfig) -> None:
    """
    Run community detection independently for each filtered Iran5 co-action layer.

    :param config: Iran5 pipeline configuration.
    :return: None. Community outputs are saved by the community managers.
    """
    ca_by_name = {ca.get_co_action(): ca for ca in config.list_ca}
    final_filter = config.co_action_filters["final"]

    for algorithm, parameters_list in config.single_layer_algorithm_dict.items():
        for parameters in parameters_list:
            cda = build_cda(algorithm, parameters)
            for co_action in config.co_action_list:
                single_layer_ca = [ca_by_name[co_action]]
                single_layer_filter = {co_action: final_filter[co_action]}

                cdm = CommunityDetectionManager(
                    config.dataset_name,
                    config.user_fraction,
                    config.type_filter,
                    config.tw,
                    single_layer_ca,
                    single_layer_filter,
                    cda,
                )
                cdm.compute_community_detection()

                chm = CharacterizationManager(
                    config.dataset_name,
                    config.user_fraction,
                    config.type_filter,
                    config.tw,
                    single_layer_ca,
                    single_layer_filter,
                    cda,
                )
                chm.compute_community_summary_statistics()
                chm.compute_metrics_communities(70)
                chm.compute_network_node_metrics(metrics=config.metrics_node_to_compute)
                chm.validate_communities()
                chm.compute_community_edge_weight_statistics()


def run_multiplex_community_detection(config: PipelineConfig) -> None:
    """
    Run Iran5 multiplex community detection and characterization.

    :param config: Iran5 pipeline configuration.
    :return: None. Community outputs are saved by the community managers.
    """
    final_filter = config.co_action_filters["final"]

    for algorithm, parameters_list in config.multiplex_algorithm_dict.items():
        for parameters in parameters_list:
            cda = build_cda(algorithm, parameters)
            cdm = CommunityDetectionManager(
                config.dataset_name,
                config.user_fraction,
                config.type_filter,
                config.tw,
                config.list_ca,
                final_filter,
                cda,
            )
            cdm.compute_community_detection()

            chm = CharacterizationManager(
                config.dataset_name,
                config.user_fraction,
                config.type_filter,
                config.tw,
                config.list_ca,
                final_filter,
                cda,
            )
            chm.compute_multiplex_community_membership_summary()
            chm.compute_community_summary_statistics()
            chm.compute_network_node_metrics(metrics=config.metrics_node_to_compute)
            chm.validate_communities()
            chm.compute_community_edge_weight_statistics()


def summarize_preprocessed_outputs(paths: DatasetPaths, ch: Checkpoint) -> None:
    """
    Log row counts for Iran5 normalized and co-action input files.

    :param paths: Iran5 path bundle.
    :param ch: Checkpoint instance.
    :return: None. Summary information is written to the log.
    """
    lm = LogManager("main")
    normalized_df = ch.read_dataframe(f"{paths.temp_data}{NORMALIZED_TWEETS_FILE}", dtype=dtype)
    lm.printl(
        f"main_iran5. normalized rows={len(normalized_df)}, users={normalized_df['userId'].nunique()}, "
        f"date_min={normalized_df['created'].min()}, date_max={normalized_df['created'].max()}"
    )
    for filename in ["URL.csv", "mention.csv", "hashtag.csv", "reply.csv", "retweet.csv"]:
        df = ch.read_dataframe(f"{paths.co_action_data}{filename}", dtype=dtype)
        lm.printl(
            f"main_iran5. {filename}: rows={len(df)}, users={df['userId'].nunique()}, "
            f"objects={df['objectId'].nunique() if 'objectId' in df.columns else 0}"
        )


def run_pipeline(config: PipelineConfig) -> None:
    """
    Execute the Iran5 coordinated-behavior pipeline.

    :param config: Iran5 pipeline configuration.
    :return: None. Each manager writes its own outputs.
    """
    lm = LogManager("main")
    ch = Checkpoint()
    paths = build_paths(config.dataset_name)
    im = InputManager(config.dataset_name)

    preprocess_iran5_input(config, paths, ch, im)
    # summarize_preprocessed_outputs(paths, ch)
    # run_user_selection(config, lm)
    # compute_similarity_edges(config)
    # characterize_unfiltered_networks(config)
    # filter_by_action_count_and_analyze_weight_thresholds(config)
    # apply_final_network_filters(config)
    # characterize_final_filtered_networks(config)
    # create_final_network_artifacts(config)
    # compare_final_network_layers(config)
    # run_single_layer_community_detection(config)
    # run_multiplex_community_detection(config)


if __name__ == "__main__":
    run_pipeline(load_config(SELECTED_DATASET))
