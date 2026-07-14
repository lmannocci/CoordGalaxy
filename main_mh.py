from __future__ import annotations

from typing import Any

from InputManager import InputManager
from configs import PipelineConfig, load_config
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.pipeline_io import DatasetPaths, build_paths, read_original_file, read_temp_file


SELECTED_DATASET = "mh"
RAW_COMMENTS_FILE = "mh.csv"
NORMALIZED_COMMENTS_FILE = "comments.csv"


def preprocess_mh_input(config: PipelineConfig, paths: DatasetPaths, ch: Checkpoint, im: InputManager) -> None:
    """
    Normalize MH Reddit rows and extract comment and text co-action artifacts.

    :param config: MH pipeline configuration.
    :param paths: MH path bundle.
    :param ch: Checkpoint instance.
    :param im: InputManager configured for the MH dataset.
    :return: None. Normalized CSVs are saved in temp_data and co-action CSV files in co_action_data.
    """
    df = read_original_file(ch, paths, RAW_COMMENTS_FILE)
    im.normalize_data(df, filename=NORMALIZED_COMMENTS_FILE)

    normalized_df = read_temp_file(ch, paths, NORMALIZED_COMMENTS_FILE)
    im.extract_comment_dataset(normalized_df, "comment.csv")
    im.extract_text_dataset(normalized_df, "commentText.npy", build_embeddings=True)


def run_user_selection(config: PipelineConfig, lm: LogManager) -> None:
    """
    Optionally analyze and apply user selection for MH.

    :param config: MH pipeline configuration.
    :param lm: Log manager.
    :return: None. Selection artifacts are saved by SelectionUserManager when enabled.
    """
    from SelectionUserManager import SelectionUserManager

    for selection_fraction in config.user_selection_fractions:
        su = SelectionUserManager(config.dataset_name, selection_fraction, config.type_filter, config.co_action_list)
        su.analyze_user_selection(config.filter_dataset)

    if config.user_fraction is None:
        lm.printl("main_mh. user_fraction=None, skipping user filtering and using all users.")
        return

    su = SelectionUserManager(config.dataset_name, config.user_fraction, config.type_filter, config.co_action_list)
    su.plot_overlapping_percentage_users()
    su.plot_number_users()
    su.apply_user_selection(config.filter_dataset)


def compute_similarity_edges(config: PipelineConfig) -> None:
    """
    Compute similarity edge lists for configured MH co-actions.

    :param config: MH pipeline configuration.
    :return: None. Similarity outputs are saved by SimilarityFunctionManager.
    """
    from Objects.CoAction.CoAction import CoAction
    from SimilarityFunctionManager import SimilarityFunctionManager

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
    Characterize unfiltered MH similarity networks.

    :param config: MH pipeline configuration.
    :return: None. Characterization outputs are saved by CharacterizationManager.
    """
    from CharacterizationManager.CharacterizationManager import CharacterizationManager

    chm = CharacterizationManager(
        config.dataset_name,
        config.user_fraction,
        config.type_filter,
        config.tw,
        config.list_ca,
        config.co_action_filters["no_filter"],
    )
    chm.compute_threshold_statistics(2, 100, 2, "nAction")
    chm.plot_threshold_statistics("nAction", 10)
    chm.compute_network_metrics(config.metrics_to_compute)


def filter_by_action_count_and_analyze_weight_thresholds(config: PipelineConfig) -> None:
    """
    Filter MH networks by action count and analyze weight thresholds.

    :param config: MH pipeline configuration.
    :return: None. Filtered graphs and weight-threshold characterization outputs are saved.
    """
    from CharacterizationManager.CharacterizationManager import CharacterizationManager
    from FilterGraphManager.FilterGraphManager import FilterGraphManager

    action_filter = config.co_action_filters["n_action"]
    fm = FilterGraphManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, action_filter)
    fm.filter_graph()

    chm = CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, action_filter)
    chm.compute_threshold_statistics(0.01, 0.99, 0.01, "w_")
    chm.plot_threshold_statistics("w_", 0.01)
    chm.compute_network_metrics(config.metrics_to_compute)


def apply_final_network_filters(config: PipelineConfig) -> None:
    """
    Apply the final MH co-action filters.

    :param config: MH pipeline configuration.
    :return: None. Filtered edge lists are saved by FilterGraphManager.
    """
    from FilterGraphManager.FilterGraphManager import FilterGraphManager

    final_filter = config.co_action_filters["final"]
    fm = FilterGraphManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    fm.filter_graph()


def characterize_final_filtered_networks(config: PipelineConfig) -> None:
    """
    Compute network metrics for the final filtered MH co-action networks.

    :param config: MH pipeline configuration.
    :return: None. Network metric tables are saved by CharacterizationManager.
    """
    from CharacterizationManager.CharacterizationManager import CharacterizationManager

    final_filter = config.co_action_filters["final"]
    chm = CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    chm.compute_network_metrics(config.metrics_to_compute)


def create_final_network_artifacts(config: PipelineConfig) -> None:
    """
    Create weighted graph and Gephi artifacts for MH.

    :param config: MH pipeline configuration.
    :return: None. Network artifacts are saved by NetworkManager.
    """
    from NetworkManager import NetworkManager

    final_filter = config.co_action_filters["final"]
    nm = NetworkManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    nm.create_weighted_graph()
    nm.save_gephi_network()


def build_cda(algorithm: str, parameters: dict[str, Any] | None) -> CDAlgorithm:
    """
    Build a community-detection algorithm object from configured parameters.

    :param algorithm: Algorithm name.
    :param parameters: Parameter dictionary, or None for algorithms without parameters.
    :return: CDAlgorithm instance.
    """
    from Objects.CDAlgorithm.CDAlgorithm import CDAlgorithm

    if parameters is None:
        return CDAlgorithm(algorithm)
    return CDAlgorithm(algorithm, parameters)


def run_single_layer_community_detection(config: PipelineConfig) -> None:
    """
    Run community detection for the filtered MH co-comment layer.

    :param config: MH pipeline configuration.
    :return: None. Community outputs are saved by the community managers.
    """
    from CharacterizationManager.CharacterizationManager import CharacterizationManager
    from CommunityDetectionManager.CommunityDetectionManager import CommunityDetectionManager

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


def run_pipeline(config: PipelineConfig) -> None:
    """
    Execute the MH Reddit coordinated-behavior pipeline.

    :param config: MH pipeline configuration.
    :return: None. Each manager writes its own outputs.
    """
    lm = LogManager("main")
    ch = Checkpoint()
    paths = build_paths(config.dataset_name)
    im = InputManager(config.dataset_name)

    # preprocess_mh_input(config, paths, ch, im)
    # run_user_selection(config, lm)
    # compute_similarity_edges(config)
    # characterize_unfiltered_networks(config)
    # filter_by_action_count_and_analyze_weight_thresholds(config)
    apply_final_network_filters(config)
    characterize_final_filtered_networks(config)
    create_final_network_artifacts(config)
    run_single_layer_community_detection(config)


if __name__ == "__main__":
    run_pipeline(load_config(SELECTED_DATASET))
