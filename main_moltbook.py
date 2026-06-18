from InputManager import InputManager
from SelectionUserManager import SelectionUserManager
from SimilarityFunctionManager import SimilarityFunctionManager
from FilterGraphManager.FilterGraphManager import *
from NetworkManager import NetworkManager
from CommunityDetectionManager.CommunityDetectionManager import *
from CharacterizationManager.CharacterizationManager import *
from OverlappingCommunityManager.OverlappingCommunityManager import *
from input_config_moltbook import *
from utils.mainMethods import *
from utils.Checkpoint.Checkpoint import *


absolute_path = os.path.dirname(__file__)
results = os.path.join(absolute_path, f".{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f".{os.sep}data{os.sep}")
path_dataset = os.path.join(data_path, f".{os.sep}{dataset_name}{os.sep}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lm = LogManager('main')
    ch = Checkpoint()
    
    im = InputManager(dataset_name)

    # NORMALIZATION
    for file_prefix in ['comment','post']: #'comment', 
        df = ch.read_dataframe(f"{path_dataset}1_moltbook_{file_prefix}.csv", dtype=dtype)
        df = im.normalize_data(df, filename=f"moltbook_{file_prefix}Text.csv")

        # EXTRACT URL, TEXT, REPLY
        df = ch.read_dataframe(f"{path_dataset}moltbook_{file_prefix}Text.csv", dtype=dtype)
        url_df = im.extract_url_dataset(df, f"moltbook_{file_prefix}URL.csv", known_url, parse_urls=False)
        text_df = im.extract_text_dataset(df, f"moltbook_{file_prefix}Text.npy") # i save the embeddings in npy format
        if file_prefix == 'comment':
            df = ch.read_dataframe(f"{path_dataset}moltbook_{file_prefix}Text.csv", dtype=dtype)
            reply_df = im.extract_reply_dataset(df, f"moltbook_comment.csv")

        # CONVERSION MANAGER - DO NOT RUN IN GENERAL
        df = ch.read_dataframe(f"{path_dataset}moltbook_{file_prefix}URL.csv", dtype=dtype) 
        cm = ConversionManager()
        df = cm.compress_user_ids(df)
        ch.save_dataframe(df, f"{path_dataset}moltbook_{file_prefix}URL.csv")
        df = ch.read_dataframe(f"{path_dataset}moltbook_comment.csv", dtype=dtype) 
        if file_prefix == 'comment':
            df = cm.compress_user_ids(df)
            ch.save_dataframe(df, f"{path_dataset}moltbook_comment.csv")


    # COORDINATED BEHAVIOR BEGIN
    # ------------------------------------------------------------------------------------------------------------------

    # SIMILARITY
    for ca in list_ca:
        sm = SimilarityFunctionManager(dataset_name, user_fraction, type_filter, tw, ca, parallelize_window=70, text_similarity_threshold=text_similarity_threshold)
        sm.compute_similarity()
        sm.convert_ids_edge_list() # DO NOT RUN IN GENERAL

    # # CHARACTERIZATION NO FILTER
    chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
    chm.compute_threshold_statistics(2, 480, 5, 'nAction')
    chm.plot_threshold_statistics('nAction', 10)
    chm.select_threshold_statistics(0.04, 0.3, 0.02, False, 'nAction', 'node')
    chm.select_threshold_statistics(10000, 20000, 1000, True, 'nAction', 'node')
    chm.compute_metrics_networks(metrics_to_compute)

    # FILTER NETWORKS - nAction
    # Selected thresholds: nAction_th = {"co-comment": 2, "co-commentText": 2, "co-commentURL": 2, "co-postText": 12, "co-postURL": 5}
    for ca in list_ca:
        filter_instance = dict_ca_filter2[ca.get_co_action()]
        fm = FilterGraphManager(dataset_name, user_fraction, type_filter, tw, ca, filter_instance)
        fm.filter_graph()

    # CHARACTERIZATION ON FILTERED NETWORK with threshold on nAction, chosen by selecting about 20000 nodes on each layer
    chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter2)
    chm.compute_threshold_statistics(0.01, 0.99, 0.01, 'w_')
    chm.plot_threshold_statistics('w_', 0.01)
    chm.select_threshold_statistics(10000, 100000, 10000, True, 'w_', 'edge')
    chm.compute_metrics_networks(metrics_to_compute)


    # # FILTER NETWORKS - weight
    # Selected thresholds: nAction_th = {"co-comment": 2, "co-commentText": 2, "co-commentURL": 2, "co-postText": 12, "co-postURL": 5}
    # weight_th = {"co-comment": 0.06, "co-commentText": 0.85, "co-commentURL": 0.32, "co-postText": 0.87, "co-postURL": 0.99}
    for ca in list_ca:
        filter_instance = dict_ca_filter3[ca.get_co_action()]
        fm = FilterGraphManager(dataset_name, user_fraction, type_filter, tw, ca, filter_instance)
        fm.filter_graph()

    # # CHARACTERIZATION Compute metrics
    chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter3)
    chm.compute_metrics_networks(metrics_to_compute)

    # CREATE NETWORKS
    nm = NetworkManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter3)
    nm.create_weighted_graph()
    nm.create_weighted_multiplex_network()
    nm.save_gephi_network()

    chm.get_ML_summary()
    chm.get_ML_layer_comparison()
    chm.plot_ML_layer_comparison()

    # Selected thresholds: nAction_th = {"co-comment": 2, "co-commentText": 2, "co-commentURL": 2, "co-postText": 12, "co-postURL": 5}
    # Weight thresholds: weight_th = {"co-comment": 0.06, "co-commentText": 0.85, "co-commentURL": 0.32, "co-postText": 0.87, "co-postURL": 0.99}

    # COMMUNITY DETECTION SINGLE LAYER 
    for algorithm, parameters_list in single_layer_algorithm_dict.items():
        for param_tuple in parameters_list:
            if algorithm == 'louvain':
                cda = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
            elif algorithm == 'infomap':
                cda = CDAlgorithm("infomap")

            for co_action, filter_instance in dict_ca_filter3.items():
                dict_ca_filter = {co_action: filter_instance}
                cdm = CommunityDetectionManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda)
                cdm.compute_community_detection()
                
                chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda)
                chm.compute_statistics_communities()
                chm.compute_metrics_communities(70)
                chm.compute_node_metrics(metrics=metrics_node_to_compute)
                chm.validate_communities()
                chm.compute_coordination_communities()

    # from input_config_IORussia import * # I have to re-import it because of the previous for loops, where I overwrite dict_ca_filter ---> dict_ca_filter = {co_action: filter_instance}
    # # MULTIMODAL COMMUNITY DETECTION AND CHARACTERIZATION
    for algorithm, parameters_list in parameters_dict.items():
        for param_tuple in parameters_list:
            cda = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
            cdm = CommunityDetectionManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter3, cda)
            cdm.compute_community_detection()
            
            chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter3, cda)
            chm.compute_info_communities()
            chm.compute_statistics_communities()
            #  chm.delete_edges_visualize_multiplex_network()

            chm.compute_node_metrics(metrics=metrics_node_to_compute) # Only for flattened algorithms, not for multilayer ones
            chm.validate_communities() # for all algorithms except flat_and_weighted_sum_louvain, flat_and_weighted_sum_infomap
            chm.compute_coordination_communities()
            chm.charactrize_url_layers_communities()

    # # OVERLAPPING COMMUNITIES
    # # multicoaction / flattened network vs single layer
    # for single_algorithm in single_layer_algorithm_dict.keys(): # 'louvain', 'infomap'
    #     for algorithm, parameters_list in parameters_dict.items(): # 
    #         if single_algorithm in algorithm:   # compare louvain with glouvain, flat_sum_weighted_louvain, flat_ec_louvain, flat_nw_louvain 
    #                                             #  and infomap with glinfomap, flat_sum_weighted_infomap, flat_ec_infomap, flat_nw_infomap
    #             for param_tuple in parameters_list:
    #                 cda_x = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
    #                 chm_x = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda_x)
    #                 # single layer network
    #                 for co_action_y in co_action_list:
    #                     list_ca_y = [CoAction(co_action_y, "tfidf_cosine_similarity")]
    #                     filter_instance_y = Filter("median", weight_th[co_action_y], Filter("merge_filter_action", nAction_th[co_action_y], None))
    #                     dict_ca_filter_y = {co_action_y: filter_instance_y}
                        
    #                     if single_algorithm == 'louvain':
    #                         cda_y = CDAlgorithm("louvain", {"resolution": 1})
    #                         prefix = 'louvain_resolution_1'
    #                     elif single_algorithm == 'infomap':
    #                         cda_y = CDAlgorithm("infomap")
    #                         prefix = 'infomap'

    #                     chm_y = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca_y, dict_ca_filter_y, cda_y)
    #                     ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix, chm_x=chm_x, chm_y=chm_y)
    #                     lm.printl(f"Overlapping between {algorithm} {param_tuple} and {single_algorithm} on {co_action_y}")
    #                     ocm.compute_overlapping(save_overlapping_tensor=True, save_intersections=True)
                        
                
    # # single layer network vs single layer network
    # for single_algorithm in single_layer_algorithm_dict.keys(): # 'louvain', 'infomap'
    #     for co_action_x in co_action_list:
    #             list_ca_x = [CoAction(co_action_x, "tfidf_cosine_similarity")]
    #             filter_instance_x = Filter("median", weight_th[co_action_x],Filter("merge_filter_action", nAction_th[co_action_x], None))
    #             dict_ca_filter_y = {co_action_x: filter_instance_x}
    #             if single_algorithm == 'louvain':
    #                 cda_x = CDAlgorithm("louvain", {"resolution": 1})
    #             elif single_algorithm == 'infomap':
    #                 cda_x = CDAlgorithm("infomap")
    #             chm_x = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca_x, dict_ca_filter_y, cda_x)
    #             # single layer network
    #             for co_action_y in co_action_list:
    #                 list_ca_y = [CoAction(co_action_y, "tfidf_cosine_similarity")]
    #                 filter_instance_y = Filter("median", weight_th[co_action_y], Filter("merge_filter_action", nAction_th[co_action_y], None))
    #                 dict_ca_filter_y = {co_action_y: filter_instance_y}
    #                 if single_algorithm == 'louvain':
    #                     cda_y = CDAlgorithm("louvain", {"resolution": 1})
    #                     prefix = 'louvain_resolution_1'
    #                 elif single_algorithm == 'infomap':
    #                     cda_y = CDAlgorithm("infomap")
    #                     prefix = 'infomap'
    #                 chm_y = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca_y, dict_ca_filter_y, cda_y)
    #                 ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix, chm_x=chm_x, chm_y=chm_y)
    #                 lm.printl(f"Overlapping between {single_algorithm} {co_action_x} and {single_algorithm} on {co_action_y}")
    #                 ocm.compute_overlapping(save_overlapping_tensor=True, save_intersections=True)
                    
    #                 # ocm.compute_single_layer_NMI() # community_size_th=None

    #                 #  # community_size_th=None must be put only for single_layer_NMI, which can be interesting computing on the whole sets of communities
    #                 ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix, chm_x=chm_x, chm_y=chm_y, community_size_th=70)
    #                 ocm.compute_single_layer_NMI() # community_size_th=200


    # for single_algorithm in single_layer_algorithm_dict.keys(): # 'louvain_resolution_1',
    #     if single_algorithm == 'louvain':
    #         cda = CDAlgorithm("louvain", {"resolution": 1})
    #         prefix = 'louvain_resolution_1'
    #     elif single_algorithm == 'infomap':
    #         cda = CDAlgorithm("infomap")
    #         prefix = 'infomap'

    #     ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix)
    #     ocm_th = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix, community_size_th=70)
        
    #     ocm.plot_heatmap_single_layer_NMI()
    #     ocm.combine_coordination_communities(cda) # combine the single layer coordination communities weight info (then I combine also the multimodal coordination communities in the following lines)
    #     ocm.combine_validation_communities(cda) # for all algorithms except flat_and_weighted_sum_louvain, flat_and_weighted_sum_infomap

    #     # ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix, community_size_th=70)
    #     ocm_th.plot_heatmap_single_layer_NMI()
    #     ocm_th.plot_heatmap_overlapping_matrix()
    #     for mid_th in [0.5, 0.7, 0.9]:
    #         ocm_th.plot_stacked_flux(type_aggregation='communities', mid_th=mid_th, metric='harmonicMean', plot_heatmap_list=None)
    #     ocm_th.plot_stacked_flux(type_aggregation='users', metric='absolute', plot_heatmap_list=None)

    #     # COMBINE SINGLE LAYERS METRIC COMMUNITIES AND NODES
    #     ocm_th.combine_single_layer_metrics_communities(cda)
        
    #     ocm_th.combine_node_metrics(cda) # combine the metrics of the nodes of the 5 co-actions + in the following lines the multimodal ones

    #     # combine the metrics of the nodes of the flattened networks
    #     for algorithm, parameters_list in parameters_dict.items():
    #         for param_tuple in parameters_list:
    #             cda = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
    
    #             # compare louvain with glouvain, flat_sum_weighted_louvain, flat_ec_louvain, flat_nw_louvain 
    #             #  and infomap with glinfomap, flat_sum_weighted_infomap, flat_ec_infomap, flat_nw_infomap
    #             if single_algorithm in algorithm:  
    #                 ocm_th.combine_node_metrics(cda)
    #                 ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix) # without community_size_th
    #                 ocm.combine_coordination_communities(cda)
    #                 ocm.combine_validation_communities(cda) # for all algorithms except flat_and_weighted_sum_louvain, flat_and_weighted_sum_infomap
        
    #     # ocm_th = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, prefix, community_size_th=70)
    #     # BOX PLOT METRICS LOST COMMON GAINED NODES
    #     ocm_th.plot_boxplot_metrics_gained_lost_nodes()

    #     # COMPARISON SINGLE LAYERS METRIC COMMUNITIES
    #     ocm_th.plot_single_layer_metrics('cosine_similarity')
    #     ocm_th.plot_barchart_cosine_similarity()
    #     ocm_th.plot_single_layer_metrics('umap')
    #     ocm_th.plot_single_layer_metrics('t_sne')
    #     ocm_th.plot_single_layer_metrics('pca')
    #     ocm_th.plot_single_layer_metrics('starplot', type_visualization_starplot='grid')
    #     ocm_th.plot_single_layer_metrics('starplot', type_visualization_starplot='single')
    #     ocm_th.compute_coordination_by_label()
    #     ocm_th.compute_validation_by_label()
        
    #     # PLOT MULTIMODAL COORDINATION VALIDATION (i compare only multimodal and flat_weighted_sum_louvain/infomap)
    #     for algorithm, parameters_list in parameters_dict.items():
    #         if (algorithm == 'flat_weighted_sum_louvain' or algorithm == 'flat_weighted_sum_infomap') and single_algorithm in algorithm:
    #             for param_tuple in parameters_list:
    #                 lm.printl(f"Plotting {algorithm} {param_tuple} coordination by label against {single_algorithm}")
    #                 cda = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
    #                 ocm_th.plot_validation_multimodal(cda)
    #                 ocm_th.plot_coordination_by_label(cda)





