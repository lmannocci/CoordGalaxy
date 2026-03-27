from utils.mainMethods import *
from Objects.CoAction.CoAction import *
from utils.Checkpoint.Checkpoint import *
import os
import numpy as np

class DirectoryManager:
    def __init__(self, file_name, dataset_name, data_path=None, results=None, user_fraction=None, type_filter=None, tw=None, ca=None, filter_instance=None, list_ca=None, dict_ca_filter=None, cda=None):
        """
            :param ch: [Checkpoint] Checkpoint instance to save object.
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window. The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
        """
        self.lm = LogManager('main')
        self.ch = Checkpoint()

        # InputManager
        self.dataset_name = dataset_name
        self.data_path = data_path

        # SelectionUserManager
        self.user_fraction = user_fraction
        self.type_filter = type_filter

        # SimilarityFunctionManager
        self.tw = tw
        self.ca = ca

        # FilterGraphManager
        self.filter_instance = filter_instance

        # NetworkManager, CommunityDetectionManager
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter

        # CommunityDetectionManager
        self.cda = cda

        # CharacterizationManager

        if level[file_name] == -1:  # InputManager
            self.path_dataset = f"{data_path}{self.dataset_name}{os.sep}"
            create_directory(file_name, self.path_dataset)

            self.path_data_analysis = f"{self.path_dataset}analysis{os.sep}"
            create_directory(file_name, self.path_data_analysis)

        if level[file_name] == 0:  # SelectionUserManager
            self.path_dataset = f"{data_path}{self.dataset_name}{os.sep}"
            self.path_data_analysis = f"{self.path_dataset}analysis{os.sep}"

            # create_directory results/uk2019/
            self.result_dataset = f"{results}{self.dataset_name}{os.sep}"
            create_directory(file_name, self.result_dataset)

            # create_directory results/uk2019/top_co_action_merge_0.01/
            self.path_type_filter = f"{self.result_dataset}{self.type_filter}_{self.user_fraction}{os.sep}"
            create_directory(file_name, self.path_type_filter)

        if level[file_name] >= 1 and level[file_name] <= 2: # FilterGraphManager
            self.path_dataset = f"{data_path}{self.dataset_name}{os.sep}"

            self.result_dataset = f"{results}{self.dataset_name}{os.sep}"
            create_directory(file_name, self.result_dataset)

            # results/uk2019/top_retweeters/
            self.path_type_filter = f"{self.result_dataset}{self.type_filter}_{self.user_fraction}{os.sep}"
            create_directory(file_name, self.path_type_filter)

            # results/uk2019/top_retweeters/merged_networks/
            # results/uk2019/top_retweeters/temporal_networks/
            if self.tw.get_type_output_network() == "merged":
                self.network_result = f"{self.path_type_filter}merged_network{os.sep}"
            elif self.tw.get_type_output_network() == "temporal":
                self.network_result = f"{self.path_type_filter}temporal_network{os.sep}"
            create_directory(file_name, self.network_result)

            # results/uk2019/top_retweeters/merged_networks/ATW/
            self.path_type_time_window = f"{self.network_result}{self.tw.get_type_time_window()}{os.sep}"
            create_directory(file_name, self.path_type_time_window)

            # results/uk2019/top_retweeters/merged_networks/ATW/tw_1d/
            # root, next directories considered from here
            self.path_tw = self.__get_window_path(self.path_type_time_window)
            create_directory(file_name, self.path_tw)

            # create a new directory path/info_tw/
            self.path_info_tw = f"{self.path_tw}info_tw{os.sep}"
            create_directory(file_name, self.path_info_tw)

            if self.tw.get_type_output_network() == "merged":
                # create a new directory path/average/
                self.path_root = f"{self.path_tw}{self.tw.get_type_merge()}{os.sep}"
                create_directory(file_name, self.path_root)
            elif self.tw.get_type_output_network() == "temporal":
                self.path_root = self.path_tw

            self.path_ca, self.path_ca_sf = self.__get_co_action_path(self.ca, self.path_root)

            # create a new directory path/co_retweet
            create_directory(file_name, self.path_ca)
            # create a new directory path/co_retweet/overlapping_coefficient
            create_directory(file_name, self.path_ca_sf)

            # create a new directory path/co_retweet/overlapping/edge_list/ where to save the files for the edge lists
            self.path_edge_list = f"{self.path_ca_sf}edge_list{os.sep}"
            create_directory(file_name, self.path_edge_list)

            # create a new directory path/co_retweet/overlapping/processed/ where to save the files for the edge lists
            self.path_processed = f"{self.path_ca_sf}processed{os.sep}"
            create_directory(file_name, self.path_processed)

            # in this directory, I move the files of each time window, when they are merged, it is exploited only in
            # create a new directory path/co_retweet/overlapping/edge_list/temporal/
            self.path_edge_list_temporal = f"{self.path_edge_list}temporal{os.sep}"
            create_directory(file_name, self.path_edge_list_temporal)

            # create a new directory path/co_retweet/overlapping/analysis
            self.path_NF_analysis = f"{self.path_ca_sf}analysis{os.sep}"
            create_directory(file_name, self.path_NF_analysis)

            # create a new directory path/co_retweet/overlapping/info_edge_list/
            self.path_info_edge_list = f"{self.path_ca_sf}info_edge_list{os.sep}"
            create_directory(file_name, self.path_info_edge_list)

            # in this directory, I move the files of each time window, when they are merged, it is exploited only in
            # create a new directory path/co_retweet/overlapping/info_edge_list/temporal/
            self.path_info_edge_list_temporal = f"{self.path_info_edge_list}temporal{os.sep}"
            create_directory(file_name, self.path_info_edge_list_temporal)


            # I want the path to the co-action of "overlapping_coefficient", because i want to access the info_edge_list even from the
            # tf-idf_cosine_similarity (or another similarity function). Please note, that info_edge_list are computed with "overlapping"
            # similarity, but this is not referenced here
            _, self.path_ca_overlapping = self.__get_co_action_path(CoAction(self.ca.get_co_action(), "overlapping_coefficient"), self.path_root)
            # path/co_retweet/overlapping/info_edge_list/ this is exploited for threshold action filter. I have to access
            # this information even if the similarity function is tf_idf_cosine_similarity
            self.path_overlapping_info_edge_list = f"{self.path_ca_overlapping}info_edge_list{os.sep}"
            self.path_overlapping_info_edge_list_temporal = f"{self.path_overlapping_info_edge_list}temporal{os.sep}"

            if level[file_name] == 2: # FilterGraphManager
                # there can be previous filter, or it can be the first filter
                # path_tw/th_0.7/backbone_0.001/
                # path_tw/
                self.path_previous_filter, self.path_filter = self.__get_path_previous_and_filter(self.path_ca_sf, self.filter_instance)
                self.lm.printl(self.path_previous_filter)
                self.lm.printl(self.path_filter)
                # if there is a previous filter, I want to access its files
                # if there is not a previous filter, self.path_previous_filter== path_ca_sf, to which I have already accessed,
                # because it is the original files directory (see above)
                if self.filter_instance.get_previous_filter() is not None:
                    # directory path_tw/th_0.7/edge_list/
                    self.path_previous_filter_edge_list = f"{self.path_previous_filter}edge_list{os.sep}"
                    # directory path_tw/th_0.7/edge_list/temporal/
                    self.path_previous_filter_edge_list_temporal = f"{self.path_previous_filter_edge_list}temporal{os.sep}"
                    # create a directory path_tw/th_0.7/info_edge_list/
                    self.path_previous_filter_info_edge_list = f"{self.path_previous_filter}info_edge_list{os.sep}"
                    # create a directory path_tw/th_0.7/info_edge_list/temporal/
                    self.path_previous_filter_info_edge_list_temporal = f"{self.path_previous_filter_info_edge_list}temporal{os.sep}"
                    # directory path_tw/th_0.7/processed/ where to save the files for the edge lists
                    self.path_previous_filter_processed = f"{self.path_previous_filter}processed{os.sep}"
                    # directory path_tw/th_0.7/edge_list_df/
                    self.path_previous_filter_edge_list_df = f"{self.path_previous_filter}edge_list_df{os.sep}"
                    # directory path_tw/th_0.7/graph/
                    self.path_previous_filter_graph = f"{self.path_previous_filter}graph{os.sep}"
                    # create directory path_tw/th_0.7/gephi_graph/
                    self.path_previous_gephi_graph = f"{self.path_previous_filter}gephi_graph{os.sep}"
                    # directory path_tw/th_0.7/analysis/
                    self.path_previous_analysis = f"{self.path_previous_filter}analysis{os.sep}"
                    # directory path_tw/th_0.7/community/
                    self.path_previous_community = f"{self.path_previous_filter}community{os.sep}"

                # __get_threshold_mean_std must be computed here, because it needs the references to find the edge list
                # of the previous filter. It must read the edge_list, computing the statistics. This must be done, only
                # if the type of filter is one of the list. Otherwise, the threshold is already defined.
                # In FilterGraphManager, once that DirectoryManager() has been instantiated, it is called the get_filter()
                # to obtain the filter of DirectoryManager(), which has the updated version of filter, with the threshold set
                # in the following lines. I call here this part because I could need the self.path_previous_filter_edge_list
                # then I have to call again __get_path_previous_and_filter, because self.path_filter is not correctly
                # instantiated, since until here threshold could be None
                if self.filter_instance.get_type_filter() in ['low_std', 'mean', 'high_std', 'median']:
                    threshold = self.__get_threshold_mean_std(self.filter_instance)
                    self.filter_instance.set_threshold(threshold)

                    self.path_previous_filter, self.path_filter = self.__get_path_previous_and_filter(self.path_ca_sf, self.filter_instance)

                # create a directory path_tw/th_0.7/
                create_directory(file_name, self.path_filter)

                # create a directory path_tw/th_0.7/edge_list/
                self.path_filter_edge_list = f"{self.path_filter}edge_list{os.sep}"
                create_directory(file_name, self.path_filter_edge_list)

                # create a directory path_tw/th_0.7/edge_list/temporal
                self.path_filter_edge_list_temporal = f"{self.path_filter_edge_list}temporal{os.sep}"
                create_directory(file_name, self.path_filter_edge_list_temporal)

                # create a directory path_tw/th_0.7/info_edge_list/
                self.path_filter_info_edge_list = f"{self.path_filter}info_edge_list{os.sep}"
                create_directory(file_name, self.path_filter_info_edge_list)

                # create a directory path_tw/th_0.7/info_edge_list/temporal/
                self.path_filter_info_edge_list_temporal = f"{self.path_filter_info_edge_list}temporal{os.sep}"
                create_directory(file_name, self.path_filter_info_edge_list_temporal)

                # create a new directory path_tw/th_0.7/processed/ where to save the files for the edge lists
                self.path_filter_processed = f"{self.path_filter}processed{os.sep}"
                create_directory(file_name, self.path_filter_processed)

                self.path_filter_edge_list_df = f"{self.path_filter}edge_list_df{os.sep}"
                create_directory(file_name, self.path_filter_edge_list_df)

                # create directory path_tw/th_0.7/graph/
                self.path_filter_graph = f"{self.path_filter}graph{os.sep}"
                create_directory(file_name, self.path_filter_graph)

                # create directory path_tw/th_0.7/gephi_graph/
                self.path_gephi_graph = f"{self.path_filter}gephi_graph{os.sep}"
                create_directory(file_name, self.path_gephi_graph)

                # create directory path_tw/th_0.7/analysis/
                self.path_analysis = f"{self.path_filter}analysis{os.sep}"
                create_directory(file_name, self.path_analysis)

                # create directory path_tw/th_0.7/community/
                self.path_community = f"{self.path_filter}community{os.sep}"
                create_directory(file_name, self.path_community)

        if level[file_name] >= 3: # NetworkManager (CommunityDetectionManager, CharacterizationManager)
            self.path_dataset = f"{data_path}{self.dataset_name}{os.sep}"
            self.path_data_analysis = f"{self.path_dataset}analysis{os.sep}"
            self.path_temp = f"{self.path_dataset}temp{os.sep}"

            self.result_dataset = f"{results}{self.dataset_name}{os.sep}"

            self.type_algorithm = self.__get_type_algorithm(self.list_ca)
            self.dict_path_ca = {}

            self.path_type_filter = f"{self.result_dataset}{self.type_filter}_{self.user_fraction}{os.sep}"

            # results/top_retweeters/merged_networks/
            # results/top_retweeters/temporal_networks/
            if self.tw.get_type_output_network() == "merged":
                self.network_result = f"{self.path_type_filter}merged_network{os.sep}"
            elif self.tw.get_type_output_network() == "temporal":
                self.network_result = f"{self.path_type_filter}temporal_network{os.sep}"

            # results/top_retweeters/merged_networks/ATW/
            self.path_type_time_window = f"{self.network_result}{self.tw.get_type_time_window()}{os.sep}"

            # results/top_retweeters/merged_networks/ATW/tw_1d/
            # root, next directories considered from here
            self.path_tw = self.__get_window_path(self.path_type_time_window)

            if self.tw.get_type_output_network() == "merged":
                # create a new directory path/average/
                self.path_root = f"{self.path_tw}{self.tw.get_type_merge()}{os.sep}"
            elif self.tw.get_type_output_network() == "temporal":
                self.path_root = self.path_tw

            for ca in self.list_ca:
                type_ca = ca.get_co_action()
                filter = self.dict_ca_filter[type_ca]

                # path_tw/co_retweet/
                # path_tw/co_retweet/overlapping/
                path_ca, path_ca_sf = self.__get_co_action_path(ca, self.path_root)

                self.dict_path_ca[type_ca] = {}

                # path_tw/co_retweet/overlapping/edge_list/
                path_edge_list = f"{path_ca_sf}edge_list{os.sep}"
                self.dict_path_ca[type_ca]["path_NF_edge_list"] = path_edge_list

                # path_tw/co_retweet/overlapping/edge_list/temporal/
                path_edge_list_temporal = f"{path_edge_list}temporal{os.sep}"
                self.dict_path_ca[type_ca]["path_NF_edge_list_temporal"] = path_edge_list_temporal

                # path_tw/co_retweet/overlapping/processed/
                path_processed = f"{path_ca_sf}processed{os.sep}"
                self.dict_path_ca[type_ca]["path_NF_processed"] = path_processed

                # path_tw/co_retweet/overlapping/analysis/
                path_NF_analysis = f"{path_ca_sf}analysis{os.sep}"
                self.dict_path_ca[type_ca]["path_NF_analysis"] = path_NF_analysis

                # path_tw/co_retweet/overlapping/info_edge_list/
                path_NF_info_edge_list = f"{path_ca_sf}info_edge_list{os.sep}"
                self.dict_path_ca[type_ca]["path_NF_info_edge_list"] = path_NF_info_edge_list

                # path_tw/co_retweet/overlapping/info_edge_list/temporal/
                path_NF_info_edge_list_temporal = f"{path_NF_info_edge_list}temporal{os.sep}"
                self.dict_path_ca[type_ca]["path_NF_info_edge_list_temporal"] = path_NF_info_edge_list_temporal

                # get the tree of multiple filters
                _, path_filter = self.__get_path_previous_and_filter(path_ca_sf, filter)
                # a Filter can be None only from CharacterizationManager, if I want to characterize a not filtered network
                if path_filter is not None:
                    # path_tw/co_retweet/overlapping/th_0.7/backbone_0.002/
                    self.dict_path_ca[type_ca]["path_filter"] = path_filter

                    # path_tw/co_retweet/overlapping/th_0.7/graph/
                    path_filter_graph = f"{path_filter}graph{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_graph"] = path_filter_graph

                    # path_tw/co_retweet/overlapping/th_0.7/gephi_graph/
                    path_gephi_graph = f"{path_filter}gephi_graph{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_gephi_graph"] = path_gephi_graph
                    # create directory ./louvain_res_7/gephi_graph/

                    # path_tw/co_retweet/overlapping/th_0.7/edge_list/
                    path_filter_edge_list = f"{path_filter}edge_list{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_edge_list"] = path_filter_edge_list

                    # path_tw/co_retweet/overlapping/th_0.7/processed/
                    path_filter_processed = f"{path_filter}processed{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_processed"] = path_filter_processed

                    path_filter_edge_list_df = f"{path_filter}edge_list_df{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_edge_list_df"] = path_filter_edge_list_df

                    # path_tw/co_retweet/overlapping/th_0.7/analysis/
                    path_filter_analysis = f"{path_filter}analysis{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_analysis"] = path_filter_analysis

                    # path_tw/co_retweet/overlapping/th_0.7/community/
                    path_filter_community = f"{path_filter}community{os.sep}"
                    self.dict_path_ca[type_ca]["path_filter_community"] = path_filter_community

                # I want the path to the co-action of "overlapping_coefficient", because i want to access the info_edge_list even from the
                # tf-idf_cosine_similarity (or another similarity function). Please note, that info_edge_list are computed with "overlapping"
                # similarity, but this is not referenced here
                _, path_ca_overlapping = self.__get_co_action_path(CoAction(ca.get_co_action(), "overlapping_coefficient"), self.path_root)
                self.dict_path_ca[type_ca]["path_ca_overlapping"] = path_ca_overlapping
                # path/co_retweet/overlapping/info_edge_list/ this is exploited for threshold action filter. I have to access
                # this information even if the similarity function is tf_idf_cosine_similarity
                path_overlapping_info_edge_list = f"{path_ca_overlapping}info_edge_list{os.sep}"
                self.dict_path_ca[type_ca]["path_overlapping_info_edge_list"] = path_overlapping_info_edge_list
                path_overlapping_info_edge_list_temporal = f"{path_overlapping_info_edge_list}temporal{os.sep}"
                self.dict_path_ca[type_ca]["path_overlapping_info_edge_list_temporal"] = path_overlapping_info_edge_list_temporal

            if self.type_algorithm == "multi-layer":
                # create directory path_tw/multi_co_action/
                self.path_multi_co_action = f"{self.path_root}multi_co_action{os.sep}"
                create_directory(file_name, self.path_multi_co_action)

                # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/
                self.path_multi_co_action_instance = self.__get_path_multi_co_action(self.list_ca, self.dict_ca_filter)
                create_directory(file_name, self.path_multi_co_action_instance)

                # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/graph
                self.path_multi_graph = f"{self.path_multi_co_action_instance}graph{os.sep}"
                create_directory(file_name, self.path_multi_graph)

                # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/edge_list_df
                self.path_multi_edge_list_df = f"{self.path_multi_co_action_instance}edge_list_df{os.sep}"
                create_directory(file_name, self.path_multi_edge_list_df)

                # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/analysis
                self.path_analysis = f"{self.path_multi_co_action_instance}analysis{os.sep}"
                create_directory(file_name, self.path_analysis)

                # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/processed
                self.path_processed = f"{self.path_multi_co_action_instance}processed{os.sep}"
                create_directory(file_name, self.path_processed)

                # CASE multi-layer: path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/community
                self.path_community = f"{self.path_multi_co_action_instance}community{os.sep}"
                create_directory(file_name, self.path_community)

                # multi-layer: path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/visualization
                self.path_visualization = f"{self.path_multi_co_action_instance}visualization{os.sep}"
                create_directory(file_name, self.path_visualization)

                # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/analysis
                self.path_overlapping_analysis = f"{self.path_multi_co_action_instance}overlapping_analysis{os.sep}"
                create_directory(file_name, self.path_overlapping_analysis)
            elif self.type_algorithm == "one-layer":
                # in case of single layer, the co-action is only one and there is only one key in dictionary. Results will be stored here
                self.path_community = list(self.dict_path_ca.values())[0]['path_filter_community']

        if level[file_name] >= 4: # CommunityDetectionManager (CharacterizationManager)
            # according to the algorithm and parameters, the name of the directory changes
            # the procedure is in common both to one-layer and multi-layer networks
            # create directory path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/louvain_res_7/
            # create directory path_tw/co_retweet/overlapping/th_0.7/community/louvain_res_7/
            self.path_algorithm = self.__get_path_algorithm(self.path_community, self.cda)
            create_directory(file_name, self.path_algorithm)

            # create directory ./louvain_res_7/coms/
            self.path_coms = f"{self.path_algorithm}coms{os.sep}"
            create_directory(file_name, self.path_coms)

            self.path_community_graph = f"{self.path_algorithm}graph{os.sep}"
            # create directory ./louvain_res_7/graph/
            create_directory(file_name, self.path_community_graph)

            self.path_community_gephi_graph = f"{self.path_algorithm}gephi_graph{os.sep}"
            # create directory ./louvain_res_7/gephi_graph/
            create_directory(file_name, self.path_community_gephi_graph)

            # dataframe directory is used to save a df with userId, group, mapping the user to the community
            self.path_user_dataframe = f"{self.path_algorithm}user_dataframe{os.sep}"
            # create directory ./louvain_res_7/user_dataframe/
            create_directory(file_name, self.path_user_dataframe)

            # create directory ./louvain_res_7/visualization/
            self.path_community_visualization = f"{self.path_algorithm}visualization{os.sep}"
            create_directory(file_name, self.path_community_visualization)

            # multi-layer: path_tw/multi_co_action/co_retweet_overlapping_th_0.7_co_url_overlapping_th_0.7/community/analysis
            self.path_community_analysis = f"{self.path_algorithm}analysis{os.sep}"
            create_directory(file_name, self.path_community_analysis)
        if level[file_name] >= 5: # CharacterizationManager
            pass
        if level[file_name] == 6: # OverlappingCommunityManager
            self.path_overlapping_heatmap = f"{self.path_overlapping_analysis}heatmap{os.sep}"
            create_directory(file_name, self.path_overlapping_heatmap)

            self.path_overlapping_stacked_plot = f"{self.path_overlapping_analysis}stacked_plot{os.sep}"
            create_directory(file_name, self.path_overlapping_stacked_plot)

            self.path_overlapping_flux_df = f"{self.path_overlapping_stacked_plot}flux_df{os.sep}"
            create_directory(file_name, self.path_overlapping_flux_df)

            self.path_overlapping_t_sne_plot = f"{self.path_overlapping_analysis}t_sne_plot{os.sep}"
            create_directory(file_name, self.path_overlapping_t_sne_plot)

            self.path_overlapping_umap_plot = f"{self.path_overlapping_analysis}umap_plot{os.sep}"
            create_directory(file_name, self.path_overlapping_umap_plot)

            self.path_overlapping_starplot = f"{self.path_overlapping_analysis}starplot{os.sep}"
            create_directory(file_name, self.path_overlapping_starplot)

            self.path_overlapping_pca_plot = f"{self.path_overlapping_analysis}pca_plot{os.sep}"
            create_directory(file_name, self.path_overlapping_pca_plot)

            self.path_overlapping_NMI = f"{self.path_overlapping_analysis}NMI{os.sep}"
            create_directory(file_name, self.path_overlapping_NMI)

            self.path_overlapping_node_metrics_gained_lost= f"{self.path_overlapping_analysis}node_metrics_gained_lost{os.sep}"
            create_directory(file_name, self.path_overlapping_node_metrics_gained_lost)

            self.path_overlapping_KDE_plot = f"{self.path_overlapping_node_metrics_gained_lost}KDE_plot{os.sep}"
            create_directory(file_name, self.path_overlapping_KDE_plot)

            self.path_overlapping_distribution_plot = f"{self.path_overlapping_node_metrics_gained_lost}distribution_plot{os.sep}"
            create_directory(file_name, self.path_overlapping_distribution_plot)

            self.path_node_metrics_boxplot = f"{self.path_overlapping_analysis}node_metrics_boxplot{os.sep}"
            create_directory(file_name, self.path_node_metrics_boxplot)

            self.path_validation = f"{self.path_overlapping_analysis}validation{os.sep}"
            create_directory(file_name, self.path_validation)

            self.path_cosine_similarity= f"{self.path_overlapping_analysis}cosine_similarity{os.sep}"
            create_directory(file_name, self.path_cosine_similarity)

    def __get_window_path(self, path_type_time_window):
        """
            :return: [str] Return the path of the directory according to the parameters. ATW is based only on tw_str parameter,
            while OTW both on tw_str and tw_slide_interval_str. Instead, ANY implies no window at all, so it is free parameter.
        """
        if self.tw.get_type_time_window() == 'ATW':
            path_tw = f"{path_type_time_window}tw_{self.tw.get_tw_str()}{os.sep}"
        elif self.tw.get_type_time_window() == 'OTW':
            path_tw = f"{path_type_time_window}tw_{self.tw.get_tw_str()}-tw_slide_interval_{self.tw.get_tw_slide_interval_str()}{os.sep}"
        elif self.tw.get_type_time_window() == 'ANY':
            path_tw = f"{path_type_time_window}{os.sep}"
        return path_tw

    def __get_co_action_path(self, ca, path_tw):
        """
            Get co-action path of the directory according to the parameter of the co_action.
            :param path_tw: [str] Path of the directory of the time window. Starting from this directory, the rest of the directory tree is constructed
            :return: [tuple(str, str)] Tuple including the path of the co-action,  e.g., ./co-retweet, and the path of the similarity function, e.g., retweet/overlapping/.
        """
        path_ca = f"{path_tw}{ca.get_co_action()}{os.sep}"
        path_ca_sf = f"{path_ca}{ca.get_similarity_function()}{os.sep}"
        return path_ca, path_ca_sf

    # FilterGraphManager
    # ------------------------------------------------------------------------------------------------------------------
    def __get_path_previous_and_filter(self, path_ca_sf, filter_instance):
        # self.lm.printl(f"{file_name}. __get_path_previous_and_filter start")
        # filter == None only with CharacterizationManager, if I want to characterize original network
        if filter_instance is None:
            return None, None
        previous_filter = filter_instance.get_previous_filter()
        filter_concat = ""
        while previous_filter is not None:
            filter_concat = f"{str(previous_filter)}{filter_concat}{os.sep}"
            previous_filter = previous_filter.get_previous_filter()

        path_previous_filter = f"{path_ca_sf}{filter_concat}"
        path_filter = f"{path_ca_sf}{filter_concat}{str(filter_instance)}{os.sep}"

        # self.lm.printl(f"{file_name}. __get_path_previous_and_filter completed")
        return path_previous_filter, path_filter


    def __get_threshold_mean_std(self, filter_instance):
        # self.lm.printl(f"{file_name}. get_threshold_mean_std start.")
        # if this is the first filter, I read from the original edge_list directory, otherwise from the previous filter edge list directory
        if self.filter_instance.get_previous_filter() == None:
            path_read = self.path_edge_list
        else:
            path_read = self.path_previous_filter_edge_list

        edge_list_files = [pos_csv for pos_csv in os.listdir(path_read) if pos_csv.endswith('.p')]
        list_upper_bound = []
        list_mean = []
        list_median = []
        list_lower_bound = []
        for elf in edge_list_files:
            edge_list = self.ch.load_object(path_read + elf)
            weights = np.array([e[2] for e in edge_list])

            mean_value = np.mean(weights)
            std_deviation = np.std(weights)
            median_value = np.median(weights)
            upper_bound = round(mean_value + std_deviation, 2)
            lower_bound = round(mean_value - std_deviation, 2)

            list_upper_bound.append(upper_bound)
            list_lower_bound.append(lower_bound)
            list_mean.append(mean_value)
            list_median.append(median_value)

        if filter_instance.get_type_filter() =="low_std":
            threshold = list_lower_bound[0]
        elif filter_instance.get_type_filter() =="mean":
            threshold = list_mean[0]
        elif filter_instance.get_type_filter() =="high_std":
            threshold = list_upper_bound[0]
        elif filter_instance.get_type_filter() == "median":
            threshold = list_median[0]

        threshold = round(threshold, 3)
        # self.lm.printl(f"{file_name}. get_threshold_mean_std completed. Selected threshold: {str(threshold)}")
        return threshold

    # NetworkManager
    # ------------------------------------------------------------------------------------------------------------------
    def __get_type_algorithm(self, list_ca):
        if len(list_ca) == 0:
            m = "ValueError: Select at least one co-action."
            self.lm.printl(m)
            raise ValueError(m)
        elif len(list_ca) == 1:
            type_algorithm = "one-layer"
        elif len(self.list_ca) > 1:
            type_algorithm = "multi-layer"
        return type_algorithm


    def __get_path_multi_co_action(self, list_ca, dict_ca_filter):
        path = f"{self.path_multi_co_action}"
        for i, ca_object in enumerate(list_ca):
            ca_type = ca_object.get_co_action()
            similarity_function_ca = ca_object.get_similarity_function()
            filter_ca = dict_ca_filter[ca_type] # string representation of the filter, including the previous filters

            if filter_ca is not None:
                # abbreviated string format for filter
                filter_ca_str_abbr = filter_ca.filter_repr_abbr()
                text_path_ca = f"{co_action_abbreviation_map[ca_type]}_{similarity_function_map[similarity_function_ca]}_{filter_ca_str_abbr}"
            else:
                text_path_ca = f"{co_action_abbreviation_map[ca_type]}_{similarity_function_map[similarity_function_ca]}"

            if i == 0:
                path = f"{path}{text_path_ca}"
            else:
                path = f"{path}__{text_path_ca}"
        path = f"{path}{os.sep}"
        return path


    # def __get_complete_path_root_filter(self, path_ca_sf, filter):
    #     self.lm.printl(f"{file_name}. __get_complete_path_root_filter start")
    #     # filter == None only with CharacterizationManager, if I want to characterize original network
    #     if filter == None:
    #         return None
    #     previous_filter = filter.get_previous_filter()
    #     path_tree_filter = f"{path_ca_sf}"
    #     while previous_filter is not None:
    #         path_tree_filter = f"{path_tree_filter}{str(previous_filter)}{os.sep}"
    #         previous_filter = previous_filter.get_previous_filter()
    #     path_tree_filter = f"{path_ca_sf}{str(filter)}{os.sep}"
    #     self.lm.printl(f"{file_name}. __get_complete_path_root_filter completed")
    #     return path_tree_filter


    # CommunityDetectionManager
    # ------------------------------------------------------------------------------------------------------------------
    def __get_path_algorithm(self, path_community, cda):
        path = path_community + repr(cda) + os.sep
        return path

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    def get_filter(self):
        return self.filter_instance

    def get_type_algorithm(self):
        return self.type_algorithm
    
    def update_data_path(self, prefix_path):
        self.data_path =  f"{prefix_path}{self.data_path}"