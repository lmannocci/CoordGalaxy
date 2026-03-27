from InputManager import InputManager
from SelectionUserManager import SelectionUserManager
from SimilarityFunctionManager import SimilarityFunctionManager
from FilterGraphManager.FilterGraphManager import *
from NetworkManager import NetworkManager
# from Objects.TimeWindow.TimeWindow import *
# from Objects.CDAlgorithm.CDAlgorithm import *
from CommunityDetectionManager.CommunityDetectionManager import *
from CharacterizationManager.CharacterizationManager import *
from OverlappingCommunityManager.OverlappingCommunityManager import *
# from inputConfig import *
# from input_config_uk19 import *
from input_config_ira import *
from utils.mainMethods import *
from utils.Checkpoint.Checkpoint import *


absolute_path = os.path.dirname(__file__)
results = os.path.join(absolute_path, f".{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f".{os.sep}data{os.sep}")
path_dataset = os.path.join(data_path, f".{os.sep}{dataset_name}{os.sep}")
# time_range_from = "Tue Nov 12 00:00:00 +0000 2019"  # "Fri Oct 02 00:00:00 +0000 2020" già fatto da Sere
# time_range_to = "Thu Dec 12 23:59:59 +0000 2019"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ray.init()
    lm = LogManager('main')
    ch = Checkpoint()
    
    # DOWNLOAD DATA FROM ELASTICSEARCH
    im = InputManager(dataset_name)
    # read directly the csv file with all tweets, but not normalized
    df = ch.read_dataframe(f"{path_dataset}1_ira_tweets.csv", dtype=dtype)
    # rename column and other operations
    df = im.normalize_data(df, filename="2_ira_normalized_tweets.csv")
    df = ch.read_dataframe(f"{path_dataset}2_ira_normalized_tweets.csv", dtype=dtype)

    # EXTARCT URL HASHTAG AND MENTION
    url_df = im.extract_url_dataset(df, f"2_ira_normalized_URL.csv", known_url)
    hashtag_df = im.extract_hashtag_dataset(df, f"2_ira_normalized_hashtag.csv")
    mention_df = im.extract_mention_dataset(df, f"2_ira_normalized_mention.csv")

    # EXTRACT RETWEET, REPLY
    # df = ch.read_dataframe(f"{path_dataset}2_uk_normalized_info_tweets.csv", dtype=dtype)
    # retweet_df = im.extract_retweet_dataset(df, "2_uk_normalized_retweet.csv")
    # reply_df = im.extract_reply_dataset(df, "2_uk_normalized_reply.csv")

    # FILTER URL, HASHTAG, MENTION
    # url_df = ch.read_dataframe(f"{path_dataset}2_ira_normalized_url.csv", dtype=dtype)
    # url_filtered = im.filter_content_df(url_df, co_action_column['co-url-domain'], excludeDomainList, filename=f"2_ira_normalized_url_filtered.csv")
    # hashtag_df = ch.read_dataframe(f"{path_dataset}2_ira_normalized_hashtag.csv", dtype=dtype)
    # hashtag_filtered = im.filter_content_df(hashtag_df, co_action_column['co-hashtag'], excludeHashtagList, filename="2_ira_normalized_hashtag_filtered.csv")
    # mention_df = ch.read_dataframe(f"{path_dataset}2_ira_normalized_mention.csv", dtype=dtype)
    # mention_filtered = im.filter_content_df(mention_df, co_action_column['co-mention'], excludeMentionList, filename="2_ira_normalized_mention_filtered.csv")

    # NORMALIZE USERS
    # df = ch.read_dataframe(f"{path_dataset}1_ira_users.csv", dtype=dtype, line_terminator='\n')
    # im.normalize_data_user(df, "2_ira_normalized_users.csv")

    # COORDINATED BEHAVIOR BEGIN
    # ------------------------------------------------------------------------------------------------------------------
    # SELECT INITIAL USERS
    # for user_fraction in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    #     su = SelectionUserManager(dataset_name, user_fraction, type_filter, list_ca)
    #     fu = su.filter_users(save_dataset=False)

    # su = SelectionUserManager(dataset_name, user_fraction, type_filter)
    # su.plot_overlapping_percentage_users()
    # su.plot_number_users()
    # fu = su.filter_users()

    # SIMILARITY
    # for co_action in co_action_list:
    #     ca = CoAction(co_action, similarity_function)
    #     sm = SimilarityFunctionManager(dataset_name, user_fraction, type_filter, tw, ca, parallelize_window=70)
    #     sm.compute_similarity()

    # CHARACTERIZATION NO FILTER
    # chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
    # chm.compute_threshold_statistics(2, 149, 1, 'nAction')
    # chm.plot_threshold_statistics('nAction', 10)
    # chm.select_threshold_statistics(0.04, 0.3, 0.02, False, 'nAction', 'node')
    # chm.select_threshold_statistics(10000, 20000, 1000, True, 'nAction', 'node')
    # chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
    # chm.compute_metrics_networks(metrics_to_compute)

    # FILTER NETWORKS - nAction
    # nAction_th = {"co-retweet": 26, "co-reply": 4, "co-url-domain": 3, "co-mention": 48, "co-hashtag": 14}
    # for co_action in co_action_list:
    #     ca = CoAction(co_action, similarity_function)
    #     filter_instance = Filter("merge_filter_action", nAction_th[co_action], None)
    #     fm = FilterGraphManager(dataset_name, user_fraction, type_filter, tw, ca, filter_instance)
    #     fm.filter_graph()

    # CHARACTERIZATION ON FILTERED NETWORK with threshold on nAction, chosen by selecting about 20000 nodes on each layer
    # chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
    # chm.compute_threshold_statistics(0.01, 0.99, 0.01, 'weight')
    # chm.plot_threshold_statistics('weight', 0.01)
    # chm.select_threshold_statistics(500000, 1500000, 100000, True, 'weight', 'edge')


    # FILTER NETWORKS - weight
    # nAction_th = {"co-retweet": 26, "co-reply": 4, "co-url-domain": 3, "co-mention": 48, "co-hashtag": 14}
    # weight_th = {"co-retweet": 0.14, "co-reply": 0.43, "co-url-domain": 0.88, "co-mention": 0.16, "co-hashtag": 0.25}
    # for co_action in co_action_list:
    #     ca = CoAction(co_action, similarity_function)
    #     # filter_instance = Filter("th", weight_th[co_action], Filter("merge_filter_action", nAction_th[co_action], None))
    #     filter_instance = Filter("median", None, Filter("merge_filter_action", nAction_th[co_action], None))
    #     fm = FilterGraphManager(dataset_name, user_fraction, type_filter, tw, ca, filter_instance)
    #     fm.filter_graph()

    # CHARACTERIZATION Compute metrics
    chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
    # chm.compute_metrics_networks(metrics_to_compute)

    # CREATE NETWORKS
    # nm = NetworkManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
    # nm.create_weighted_graph()
    # nm.create_weighted_multiplex_network()
    # nm.save_gephi_network()

    # chm.get_ML_summary()
    # chm.get_ML_layer_comparison()
    # chm.plot_ML_layer_comparison()

    # Filter 2.3
    nAction_th = {"co-retweet": 26, "co-reply": 4, "co-url-domain": 3, "co-mention": 48, "co-hashtag": 14}
    weight_md = {"co-retweet": 0.126, "co-reply": 0.451, "co-url-domain": 0.791, "co-mention": 0.111, "co-hashtag": 0.246}
    # COMMUNITY DETECTION SINGLE LAYER
    # for co_action in co_action_list:
    #     ca = CoAction(co_action, similarity_function)
    #     list_ca = [CoAction(co_action, "tfidf_cosine_similarity")]
    #     filter_instance = Filter("median", weight_md[co_action], Filter("merge_filter_action", nAction_th[co_action], None))
    #     dict_ca_filter = {co_action: filter_instance}
    #     cda = CDAlgorithm("louvain", {"resolution": 1})

        # cdm = CommunityDetectionManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda)
    #     # cdm.compute_community_detection()
    #     chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda)
    #     # chm.compute_statistics_communities()
    #     # chm.delete_small_communities_single_layer(100)
    #     # chm.compute_metrics_communities(200)
    #     chm.compute_metrics_nodes_communities(metrics=metrics_node_to_compute, merge_existing=True) WRONG
    #     chm.compute_node_metrics(metrics=metrics_node_to_compute)



    # COMMUNITY DETECTION

    # COMMUNITY DETECTION AND CHARACTERIZATION
    # for algorithm, parameters_list in parameters_dict.items():
    #     for param_tuple in parameters_list:
    #         cda = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
    #         cdm = CommunityDetectionManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda)
            # cdm.compute_community_detection()
            # chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda)
            # chm.compute_info_communities()
            # chm.compute_statistics_communities()
            # chm.delete_edges_visualize_multiplex_network()
            # chm.compute_metrics_nodes_communities(metrics=metrics_node_to_compute, merge_existing=True) WRONG
            # chm.compute_node_metrics(metrics=metrics_node_to_compute)


    # OVERLAPPING COMMUNITIES
    # multicoaction / flattened network vs single layer
    # for algorithm, parameters_list in parameters_dict.items():
    #     for param_tuple in parameters_list:
    #         cda_x = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
    #         chm_x = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda_x)
    #         # single layer network
    #         for co_action_y in co_action_list:
    #             list_ca_y = [CoAction(co_action_y, "tfidf_cosine_similarity")]
    #             filter_instance_y = Filter("median", weight_md[co_action_y], Filter("merge_filter_action", nAction_th[co_action_y], None))
    #             dict_ca_filter_y = {co_action_y: filter_instance_y}
    #             cda_y = CDAlgorithm("louvain", {"resolution": 1})
    #             chm_y = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca_y, dict_ca_filter_y, cda_y)
    #             ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, 'louvain_resolution_1', chm_x=chm_x, chm_y=chm_y)
    #             ocm.compute_overlapping(save_overlapping_tensor=False, save_intersections=True)
                # ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter,
                #                                   'louvain_resolution_1', chm_x=chm_x, chm_y=chm_y, community_size_th=200)
    #             ocm.plot_node_metrics_gained_lost(metrics_node_to_compute) WRONG

    # single layer network vs single layer network
    # for co_action_x in co_action_list:
    #     list_ca_x = [CoAction(co_action_x, "tfidf_cosine_similarity")]
    #     filter_instance_x = Filter("median", weight_md[co_action_x],Filter("merge_filter_action", nAction_th[co_action_x], None))
    #     dict_ca_filter_y = {co_action_x: filter_instance_x}
    #     cda_x = CDAlgorithm("louvain", {"resolution": 1})
    #     chm_x = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca_x, dict_ca_filter_y, cda_x)
    #     # single layer network
    #     for co_action_y in co_action_list:
    #         list_ca_y = [CoAction(co_action_y, "tfidf_cosine_similarity")]
    #         filter_instance_y = Filter("median", weight_md[co_action_y], Filter("merge_filter_action", nAction_th[co_action_y], None))
    #         dict_ca_filter_y = {co_action_y: filter_instance_y}
    #         cda_y = CDAlgorithm("louvain", {"resolution": 1})
    #         chm_y = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca_y, dict_ca_filter_y, cda_y)
    #         ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, 'louvain_resolution_1', chm_x=chm_x, chm_y=chm_y, community_size_th=200)
    #         ocm.compute_overlapping()
    #           ocm.compute_single_layer_NMI() # community_size_th=200

    # community_size_th=None must be put only for single_layer_NMI, which can be interesting computing on the whole sets of communities
    # ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, 'louvain_resolution_1', chm_x=chm_x, chm_y=chm_y)
    #         ocm.compute_single_layer_NMI() # community_size_th=None

    # ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter,'louvain_resolution_1')
    # ocm.plot_heatmap_single_layer_NMI()

    ocm = OverlappingCommunityManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, 'louvain_resolution_1', community_size_th=200)
    ocm.plot_heatmap_overlapping_matrix()
    # for mid_th in [0.5, 0.7, 0.9]:
    #     ocm.plot_stacked_flux(type_aggregation='communities', mid_th=mid_th, metric='harmonicMean', plot_heatmap_list=None)
    # ocm.plot_stacked_flux(type_aggregation='users', metric='absolute', plot_heatmap_list=None)

    # COMBINE SINGLE LAYERS METRIC COMMUNITIES AND NODES
    # cda = CDAlgorithm("louvain", {"resolution": 1})
    # ocm.combine_single_layer_metrics_communities(cda)
    # ocm.combine_node_metrics(cda) # combine the metrics of the nodes of the 5 co-actions + in the following lines
    # combine the metrics of the nodes of the flattened networks
    # for algorithm, parameters_list in parameters_dict.items(): # only flat_sum_weighted_louvain
    #     for param_tuple in parameters_list:
    #         cda = CDAlgorithm(algorithm, get_algorithm_param(algorithm, param_tuple))
    #         ocm.combine_node_metrics(cda)

    # BOX PLOT METRICS LOST COMMON GAINED NODES
    # ocm.plot_boxplot_metrics_gained_lost_nodes()

    # COMPARISON SINGLE LAYERS METRIC COMMUNITIES
    # ocm.plot_heatmap_single_layer_NMI()
    # ocm.plot_single_layer_metrics('cosine_similarity')
    # ocm.plot_single_layer_metrics('umap')
    # ocm.plot_single_layer_metrics('t_sne')
    # ocm.plot_single_layer_metrics('pca')
    # ocm.plot_single_layer_metrics('starplot', type_visualization_starplot='grid')
    # ocm.plot_single_layer_metrics('starplot', type_visualization_starplot='single')




