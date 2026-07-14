from __future__ import annotations

from typing import Any

from CharacterizationManager.CharacterizationManager import CharacterizationManager
from CommunityDetectionManager.CommunityDetectionManager import CommunityDetectionManager
from FilterGraphManager.FilterGraphManager import FilterGraphManager
from InputManager import InputManager
from LatexTableManager import LatexTableManager
from NetworkManager import NetworkManager
from Objects.CDAlgorithm.CDAlgorithm import CDAlgorithm
from CommunityComparisonManager import CommunityComparisonManager
from SelectionUserManager import SelectionUserManager
from SimilarityFunctionManager import SimilarityFunctionManager
from configs import PipelineConfig, load_config
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.pipeline_io import DatasetPaths, build_paths, read_original_file, read_temp_file


SELECTED_DATASET = "moltbook"

MOLTBOOK_INPUT_FILES = {
    "comment": {
        "raw": "comments.csv",
        "normalized": "commentText.csv",
        "url": "commentURL.csv",
        "text_embedding": "commentText.npy",
    },
    "post": {
        "raw": "posts.csv",
        "normalized": "postText.csv",
        "url": "postURL.csv",
        "text_embedding": "postText.npy",
    },
}
COMMENT_CO_ACTION_FILE = "comment.csv"


def preprocess_moltbook_input(config: PipelineConfig, paths: DatasetPaths, ch: Checkpoint, im: InputManager) -> None:
    """
    Normalize Moltbook source files and extract co-action artifacts.

    :param config: Pipeline configuration.
    :param paths: Moltbook path bundle.
    :param ch: Checkpoint instance.
    :param im: InputManager configured for the Moltbook dataset.
    :return: None. Normalized CSVs are saved in temp_data and co-action CSV/NPY files in co_action_data.
    """
    for file_type, filenames in MOLTBOOK_INPUT_FILES.items():
        df = read_original_file(ch, paths, filenames["raw"])
        im.normalize_data(df, filename=filenames["normalized"])

        normalized_df = read_temp_file(ch, paths, filenames["normalized"])
        im.extract_url_dataset(normalized_df, filenames["url"], config.known_url, parse_urls=False)
        im.extract_text_dataset(normalized_df, filenames["text_embedding"])

        if file_type == "comment":
            im.extract_reply_dataset(normalized_df, COMMENT_CO_ACTION_FILE)


def run_user_selection(config: PipelineConfig, lm: LogManager) -> None:
    """
    Analyze and optionally apply user selection.

    :param config: Pipeline configuration.
    :param lm: Log manager.
    :return: None. Selection artifacts are saved by SelectionUserManager.
    """
    for selection_fraction in config.user_selection_fractions:
        su = SelectionUserManager(config.dataset_name, selection_fraction, config.type_filter, config.co_action_list)
        su.analyze_user_selection(config.filter_dataset)

    if config.user_fraction is None:
        lm.printl("main_moltbook. user_fraction=None, skipping user filtering and using all users.")
        return

    su = SelectionUserManager(config.dataset_name, config.user_fraction, config.type_filter, config.co_action_list)
    su.plot_overlapping_percentage_users()
    su.plot_number_users()
    su.apply_user_selection(config.filter_dataset)


def compute_similarity_edges(config: PipelineConfig) -> None:
    """
    Compute similarity edge lists for all configured co-actions.

    :param config: Pipeline configuration.
    :return: None. Similarity outputs are saved by SimilarityFunctionManager.
    """
    for ca in config.list_ca:
        sm = SimilarityFunctionManager(
            config.dataset_name,
            config.user_fraction,
            config.type_filter,
            config.tw,
            ca,
            parallelize_window=config.similarity_parallelize_window,
            text_similarity_threshold=config.text_similarity_threshold,
            text_similarity_chunk_size=config.text_similarity_chunk_size,
        )
        sm.compute_similarity()


def characterize_unfiltered_networks(config: PipelineConfig) -> None:
    """
    Characterize unfiltered similarity networks.

    :param config: Pipeline configuration.
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
    Filter networks by action count and analyze weight thresholds.

    :param config: Pipeline configuration.
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
    Apply the final co-action filters.

    :param config: Pipeline configuration.
    :return: None. Filtered edge lists are saved by FilterGraphManager.
    """
    final_filter = config.co_action_filters["final"]
    fm = FilterGraphManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    fm.filter_graph()


def characterize_final_filtered_networks(config: PipelineConfig) -> None:
    """
    Compute network metrics for the final filtered co-action networks.

    :param config: Pipeline configuration.
    :return: None. Network metric tables are saved by CharacterizationManager.
    """
    final_filter = config.co_action_filters["final"]
    chm = CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    chm.compute_network_metrics(config.metrics_to_compute)


def create_final_network_artifacts(config: PipelineConfig) -> None:
    """
    Create weighted graph, multiplex graph, and Gephi artifacts.

    :param config: Pipeline configuration.
    :return: None. Network artifacts are saved by NetworkManager.
    """
    final_filter = config.co_action_filters["final"]
    nm = NetworkManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, config.list_ca, final_filter)
    nm.create_weighted_graph()
    nm.create_weighted_multiplex_network()
    nm.save_gephi_network()


def compare_final_network_layers(config: PipelineConfig) -> None:
    """
    Compute and plot multiplex layer comparison for the final network.

    :param config: Pipeline configuration.
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
    Run community detection and characterization independently for each filtered co-action layer.

    :param config: Pipeline configuration.
    :return: None. Community outputs are saved by the community managers.
    """
    ca_by_name = {ca.get_co_action(): ca for ca in config.list_ca}
    final_filter = config.co_action_filters["final"]

    for algorithm, parameters_list in config.single_layer_algorithm_dict.items():
        for parameters in parameters_list:
            cda = build_cda(algorithm, parameters)

            for co_action, filter_instance in final_filter.items():
                single_layer_ca = [ca_by_name[co_action]]
                single_layer_filter = {co_action: filter_instance}

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


def build_multiplex_characterization_manager(config: PipelineConfig, cda: CDAlgorithm) -> CharacterizationManager:
    """
    Build a CharacterizationManager for the final Moltbook multiplex network.

    :param config: Pipeline configuration.
    :param cda: Community-detection algorithm object.
    :return: CharacterizationManager configured for multiplex community outputs.
    """
    return CharacterizationManager(
        config.dataset_name,
        config.user_fraction,
        config.type_filter,
        config.tw,
        config.list_ca,
        config.co_action_filters["final"],
        cda,
    )


def run_multiplex_community_detection(config: PipelineConfig) -> None:
    """
    Run multiplex community detection.

    :param config: Pipeline configuration.
    :return: None. Community outputs are saved by CommunityDetectionManager.
    """
    for algorithm, parameters_list in config.multiplex_algorithm_dict.items():
        for parameters in parameters_list:
            cda = build_cda(algorithm, parameters)
            # final_filter = config.co_action_filters["final"]
            # cdm = CommunityDetectionManager(
            #     config.dataset_name,
            #     config.user_fraction,
            #     config.type_filter,
            #     config.tw,
            #     config.list_ca,
            #     final_filter,
            #     cda,
            # )
            # cdm.compute_community_detection()

def characterize_multiplex_communities(config: PipelineConfig, top_n: int = 10) -> None:
    """
    Characterize already-computed multiplex community outputs.

    :param config: Pipeline configuration.
    :param top_n: Number of top communities included in top-community outputs.
    :return: None. Characterization CSV outputs are saved by CharacterizationManager.
    """
    for algorithm, parameters_list in config.multiplex_algorithm_dict.items():
        for parameters in parameters_list:
            cda = build_cda(algorithm, parameters)
            chm = build_multiplex_characterization_manager(config, cda)
            # chm.compute_multiplex_community_membership_summary()
            # chm.compute_community_summary_statistics()
            # chm.compute_network_node_metrics(metrics=config.metrics_node_to_compute)
            # chm.validate_communities()
            # chm.compute_community_edge_weight_statistics()
            # chm.save_top_communities_summary(top_n=top_n)
            chm.charactrize_url_layers_communities(top_n=top_n)


def generate_latex_tables(config: PipelineConfig, top_n: int = 10) -> None:
    """
    Generate predefined LaTeX tables from characterization CSV outputs.

    :param config: Pipeline configuration.
    :param top_n: Number of top communities included in the table.
    :return: None. LaTeX files are saved in each community analysis directory.
    """
    for algorithm, parameters_list in config.multiplex_algorithm_dict.items():
        for parameters in parameters_list:
            cda = build_cda(algorithm, parameters)
            ltm = LatexTableManager(
                config.dataset_name,
                config.user_fraction,
                config.type_filter,
                config.tw,
                config.list_ca,
                config.co_action_filters["final"],
                cda,
            )
            # ltm.build_top_communities_structural_table(top_n=top_n)
            ltm.build_url_category_composition_table(top_n=top_n)




def build_single_layer_characterization_manager(
    config: PipelineConfig,
    co_action: str,
    cda: CDAlgorithm,
) -> CharacterizationManager:
    """
    Build a CharacterizationManager for one final filtered co-action layer.

    :param config: Pipeline configuration.
    :param co_action: Canonical co-action id.
    :param cda: Community-detection algorithm object.
    :return: CharacterizationManager for one single-layer community result.
    """
    list_ca = [ca for ca in config.list_ca if ca.get_co_action() == co_action]
    dict_ca_filter = {co_action: config.co_action_filters["final"][co_action]}
    return CharacterizationManager(config.dataset_name, config.user_fraction, config.type_filter, config.tw, list_ca, dict_ca_filter, cda)


def build_community_comparison_manager(
    config: PipelineConfig,
    file_prefix: str,
    community_size_th: int | None = None,
) -> CommunityComparisonManager:
    """
    Build the front-end manager used to compare community outputs.

    :param config: Pipeline configuration.
    :param file_prefix: Prefix used for comparison output filenames.
    :param community_size_th: Optional minimum community size.
    :return: CommunityComparisonManager instance.
    """
    return CommunityComparisonManager(
        config.dataset_name,
        config.user_fraction,
        config.type_filter,
        config.tw,
        config.list_ca,
        config.co_action_filters["final"],
        file_prefix,
        community_size_th=community_size_th,
    )


def compute_multiplex_single_layer_overlaps(config: PipelineConfig) -> None:
    """
    Compare each compatible multiplex/flattened community output against each single-layer output.

    :param config: Pipeline configuration.
    :return: None. Overlap tensors are saved by CommunityComparisonManager.
    """
    for single_algorithm in config.single_layer_algorithm_dict.keys():
        for algorithm, parameters_list in config.multiplex_algorithm_dict.items():
            if single_algorithm not in algorithm:
                continue
            for parameters in parameters_list:
                cda_x = build_cda(algorithm, parameters)
                chm_x = build_multiplex_characterization_manager(config, cda_x)
                for co_action_y in config.co_action_list:
                    cda_y = build_cda(single_algorithm, {"resolution": 1} if single_algorithm == "louvain" else None)
                    prefix = "louvain_resolution_1" if single_algorithm == "louvain" else "infomap"
                    chm_y = build_single_layer_characterization_manager(config, co_action_y, cda_y)
                    comparison_manager = build_community_comparison_manager(config, prefix)
                    comparison_manager.compute_overlap(
                        chm_x,
                        chm_y,
                        save_overlapping_tensor=True,
                        save_intersections=True,
                    )


def compute_single_layer_overlaps(config: PipelineConfig, community_size_th: int = 70) -> None:
    """
    Compare every pair of single-layer community outputs and compute NMI for size-filtered comparisons.

    :param config: Pipeline configuration.
    :param community_size_th: Minimum community size for NMI.
    :return: None. Overlap tensors and NMI outputs are saved by CommunityComparisonManager.
    """
    for single_algorithm in config.single_layer_algorithm_dict.keys():
        cda = build_cda(single_algorithm, {"resolution": 1} if single_algorithm == "louvain" else None)
        prefix = "louvain_resolution_1" if single_algorithm == "louvain" else "infomap"
        for co_action_x in config.co_action_list:
            chm_x = build_single_layer_characterization_manager(config, co_action_x, cda)
            for co_action_y in config.co_action_list:
                chm_y = build_single_layer_characterization_manager(config, co_action_y, cda)
                comparison_manager = build_community_comparison_manager(config, prefix)
                comparison_manager.compute_overlap(
                    chm_x,
                    chm_y,
                    save_overlapping_tensor=True,
                    save_intersections=True,
                )
                thresholded_comparison_manager = build_community_comparison_manager(config, prefix, community_size_th)
                thresholded_comparison_manager.compute_single_layer_nmi(chm_x, chm_y)


def plot_and_combine_overlapping_outputs(config: PipelineConfig, community_size_th: int = 70) -> None:
    """
    Plot and combine outputs generated by the overlapping-community comparison steps.

    :param config: Pipeline configuration.
    :param community_size_th: Minimum community size used in size-filtered plots.
    :return: None. Plots and combined CSVs are saved by CommunityComparisonManager.
    """
    for single_algorithm in config.single_layer_algorithm_dict.keys():
        cda = build_cda(single_algorithm, {"resolution": 1} if single_algorithm == "louvain" else None)
        prefix = "louvain_resolution_1" if single_algorithm == "louvain" else "infomap"
        comparison = build_community_comparison_manager(config, prefix)
        comparison_th = build_community_comparison_manager(config, prefix, community_size_th)

        comparison.plot_overlap_heatmaps()
        comparison.combine_coordination_communities(cda)
        comparison.combine_validation_communities(cda)
        comparison_th.plot_overlap_heatmaps()
        for mid_th in [0.5, 0.7, 0.9]:
            comparison_th.plot_stacked_flux(type_aggregation="communities", mid_th=mid_th, metric="harmonicMean")
        comparison_th.plot_stacked_flux(type_aggregation="users", metric="absolute")

        comparison_th.combine_single_layer_metrics_communities(cda)
        comparison_th.combine_node_metrics(cda)
        comparison_th.plot_boxplot_metrics_gained_lost_nodes()
        comparison_th.plot_single_layer_metrics("cosine_similarity")
        comparison_th.plot_barchart_cosine_similarity()
        comparison_th.plot_single_layer_metrics("umap")
        comparison_th.plot_single_layer_metrics("t_sne")
        comparison_th.plot_single_layer_metrics("pca")
        comparison_th.plot_single_layer_metrics("starplot", type_visualization_starplot="grid")
        comparison_th.plot_single_layer_metrics("starplot", type_visualization_starplot="single")
        comparison_th.compute_coordination_by_label()
        comparison_th.compute_validation_by_label()


def run_pipeline(config: PipelineConfig) -> None:
    """
    Run the Moltbook pipeline until community characterization.

    :param config: Pipeline configuration.
    :return: None. Artifacts are saved in the configured dataset result directories.
    """
    lm = LogManager("main")
    ch = Checkpoint()
    paths = build_paths(config.dataset_name)
    im = InputManager(config.dataset_name)

    # preprocess_moltbook_input(config, paths, ch, im)
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
    characterize_multiplex_communities(config, top_n=10)
    generate_latex_tables(config, top_n=10)

    
    # compute_multiplex_single_layer_overlaps(config)
    # compute_single_layer_overlaps(config, community_size_th=70)
    # plot_and_combine_overlapping_outputs(config, community_size_th=70)


if __name__ == "__main__":
    run_pipeline(load_config(SELECTED_DATASET))
