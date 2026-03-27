from utils.common_variables import *
from utils.LogManager.LogManager import *

class IntegrityConstraintManager:
    def __init__(self, file_name):
        self.lm = LogManager("main")
        self.file_name = file_name

    # TimeWindow
    # ------------------------------------------------------------------------------------------------------------------
    def check_type_output(self, type_output_network, type_merge):
        if type_output_network == "merged":
            if type_merge not in available_type_merge:
                m = f"{self.file_name}. {type_merge} is not available. Available types are: {str(available_type_merge)}."
                self.lm.printl(m)
                raise ValueError(m)

    # CoAction
    # ------------------------------------------------------------------------------------------------------------------
    def check_co_action_availability(self, co_action, similarity_function):
        """
            Check consistency co-actions and similarity function.
        """
        if co_action not in available_co_action.keys():
            m = f"Co-action {co_action} is not available."
            self.lm.printl(m)
            raise ValueError(m)
        else:
            if similarity_function not in available_co_action[co_action]:
                m = f"{self.file_name}Similarity function {similarity_function} is not available for co-action {co_action}."
                self.lm.printl(m)
                raise ValueError(m)


    # InputManager
    # ------------------------------------------------------------------------------------------------------------------
    def check_type_filter(self, type_filter):
        if type_filter not in available_type_filter:
            m = f"{self.file_name}. {type_filter} is not a valid filter type. Available filters are: {available_type_filter}."
            self.lm.printl(m)
            raise ValueError(m)

    def check_user_fraction(self, user_fraction):
        if not isinstance(user_fraction, float):
            m = f"{self.file_name}. {user_fraction} is not a float or integer."
            self.lm.printl(m)
            raise TypeError(m)

        if user_fraction < 0 or user_fraction > 1:
            m = f"{self.file_name}. {user_fraction} must be between 0 and 1."
            self.lm.printl(m)
            raise ValueError(m)

    # SimilarityFunctionManager
    # ------------------------------------------------------------------------------------------------------------------
    def check_sparse_computation(self, ca, sparse_computation, save_info, parallelize_similarity):
        """
            Check if the chosen sparse_computation is implemented for the chosen similarity function
            :param sparse_computation: [bool] If true, compute the similarity function using the sparse implementation, memorizing
            the whole similarity matrix, before discarding zero values. Instead, if false, it computes the similarity for each couples
            of user and only if nonzero value, it is saved in memory.
        """

        if sparse_computation == True and ca.get_similarity_function() not in sparse_computation_function:
            m = f"{self.file_name}. {ca.get_similarity_function()} does not have the sparse computation implemented."
            self.lm.printl(m)
            raise ValueError(m)

        if sparse_computation == False and ca.get_similarity_function() not in dense_computation_function:
            m = f"{self.file_name}. {ca.get_similarity_function()} does not have the dense computation implemented."
            self.lm.printl(m)
            raise ValueError(m)

        if sparse_computation == True and save_info == True:
            m = f"{self.file_name}. The save of info_edge_list is not implemented if sparse_computation is True."
            self.lm.printl(m)
            raise ValueError(m)

        if sparse_computation == True and parallelize_similarity == True:
            m = f"{self.file_name}. The parallelization of the similarity computation is not implemented if sparse_computation is True."
            self.lm.printl(m)
            raise ValueError(m)

    def check_list_co_action(self, co_action_list):
        for ca in co_action_list:
            if ca not in available_co_action.keys():
                m = f"{self.file_name}. {ca} is not an available co-action. Available co-actions are: {str(available_co_action.keys())}."
                self.lm.printl(m)
                raise ValueError(m)
    # FilterGraphManager
    # ------------------------------------------------------------------------------------------------------------------


    # Filter
    # ------------------------------------------------------------------------------------------------------------------
    def check_filter_graph(self, type_filter, threshold, previous_filter):
        if type_filter not in available_filter_graph:
            m = f"{self.file_name}. type_filter is {str(type_filter)}, please select a valid filter among available filters: {str(available_filter_graph)}."
            self.lm.printl(m)
            raise ValueError(m)

        if previous_filter is not None:
            # if not isinstance(previous_filter, Filter):
            #     m = f"{self.file_name}. previousFilter must be of type Filter."
            #     self.lm.printl(m)
            #     raise ValueError(m)

            if type_filter == "filter_merge_action":
                m = (f"""{self.file_name}. type_filter={type_filter}, but this type of filter can be apply only as first filter, 
                        because it performs a filter before merging the network of different time windows and the it merges the results.. Please set previous_filter=None.
                """).replace("\n", " ")
                self.lm.printl(m)
                raise ValueError(m)

        if isinstance(threshold, (int, float)) or (threshold is None and (type_filter in ['low_std', 'mean', 'high_std', 'th', 'median'])):
            if type_filter == "backbone" or type_filter == "th":
                if threshold is None or (threshold < 0 or threshold > 1):
                    m = f"{self.file_name}. Threshold is {str(threshold)}, please select a valid value fo threshold between 0 and 1."
                    self.lm.printl(m)
                    raise ValueError(m)
            elif type_filter == "threshold_action":
                if threshold is None or (threshold < 1):
                    m = f"{self.file_name}. Threshold is {str(threshold)}, please select a valid value fo threshold greater than 1."
                    self.lm.printl(m)
                    raise ValueError(m)
        else:
            m = f"{self.file_name}. Threshold is {str(threshold)}, please select a float or integer to filter edges."
            self.lm.printl(m)
            raise ValueError(m)
        
    # NetworkManager / CommunityDetectionManager
    # ------------------------------------------------------------------------------------------------------------------
    def check_co_action(self, list_ca, dict_ca_filter):
        for ca_type in dict_ca_filter.keys():
            match = False
            for ca_object in list_ca:
                if ca_object.get_co_action() == ca_type:
                    match = True
            if match == False:
                m = f"{self.file_name}. For each co_action specified in list_ca, must be specified in dict_ca_threshold, the corresponding threshold."
                self.lm.printl(m)
                raise Exception(m)

    # def check_dict_ca_filter(self, dict_ca_filter):
    #     for ca_type, filter in dict_ca_filter.items():
    #         if (level[self.file_name] == 5 and (filter is not None and not isinstance(filter, Filter))) or (level[self.file_name] !=5 and not isinstance(filter, Filter)):
    #                 m = f"{self.file_name}. For each co_action specified in list_ca, must be specified in dict_ca_threshold a valid instance of Filter(). Available filters type are: {available_filter_graph}."
    #                 self.lm.printl(m)
    #                 raise ValueError(m)

    # CDAlgorithm
    # ------------------------------------------------------------------------------------------------------------------
    def check_CDAlgorithm(self, algorithm_name, parameters):
        # check algorithm
        available_algorithms = one_layer_algorithm + multi_layer_algorithm + multi_temporal_multi_layer_algorithm
        if algorithm_name not in available_algorithms:
            m = (f"""{self.file_name}. Select an available algorithm among those available. 
                 f"one_layer_algorithm: {str(one_layer_algorithm)} 
                 f"multi_layer_algorithm: {str(multi_layer_algorithm)} 
                 multi_temporal_multi_layer_algorithm: {str(multi_temporal_multi_layer_algorithm)}""")
            self.lm.printl(m)
            raise ValueError(m)

        # required_algorithm_parameters[algorithm_name] is a list of tuples
        if parameters is None:
            parameters = {}
        for param, description in required_algorithm_parameters[algorithm_name]:
            if param not in parameters.keys():
                m = f"{self.file_name}. {param} is a required parameter for {algorithm_name}. {description}."
                self.lm.printl(m)
                raise ValueError(m)
        #
        # # check parameters
        # if algorithm_name in ["louvain"]:
        #     if "resolution" not in parameters.keys():
        #         m = f"{self.file_name}. Define parameter resolution."
        #         self.lm.printl(m)
        #         raise ValueError(m)
        # elif algorithm_name == "clique_percolation":
        #     if "m" not in parameters.keys() or "k" not in parameters.keys():
        #         m = f"{self.file_name}. Define parameter m (minimum number of actors in a clique) and k (minimum number of layers)."
        #         self.lm.printl(m)
        #         raise ValueError(m)
        # elif algorithm_name == "abacus":
        #     if "min_actors" not in parameters.keys() or "min_layers" not in parameters.keys():
        #         m = f"{self.file_name}. Define parameter min_actors and min_layers."
        #         raise ValueError(m)
        # elif algorithm_name == "glouvain":
        #     if "omega" not in parameters:
        #         m = f"{self.file_name}. Define parameter omega."
        #         self.lm.printl(m)
        #         raise ValueError(m)


    # CommunityDetectionManager
    # ------------------------------------------------------------------------------------------------------------------
    def check_type_algorithm(self, tw, list_ca, algorithm_name):
        data_m = f"""type_output_network: {tw.get_type_output_network()} - number_co_action: {str(len(list_ca))} - algorithm: {algorithm_name}."""
        self.lm.printl(algorithm_name)
        if len(list_ca) == 0:
            m = "ValueError: Select at least one co-action."
            self.lm.printl(m)
            raise ValueError(m)
        elif len(list_ca) == 1:
            if tw.get_type_output_network() == "merged":
                if algorithm_name not in one_layer_algorithm:
                    m = f"""ValueError: {data_m} The setting you give is not neither multi-layer neither temporal. Select a CDA algorithm for one-layer community detection."""
                    self.lm.printl(m)
                    raise ValueError(m)
            elif tw.get_type_output_network() == "temporal":
                self.lm.printl(f"{data_m} Multi-layer CDA algorithm exploited for multi-layer temporal analysis.")
        elif len(list_ca) > 1:
            if tw.get_type_output_network() == "merged":
                if algorithm_name not in multi_layer_algorithm:
                    m = f"ValueError: {data_m} Select a CDA algorithm for multi-layer community detection."
                    self.lm.printl(m)
                    raise ValueError(m)
            elif tw.get_type_output_network() == "temporal":
                if algorithm_name not in multi_temporal_multi_layer_algorithm:
                    m = f"ValueError: {data_m} Select a CDA algorithm for multi-layer community detection."
                    self.lm.printl(m)
                    raise ValueError(m)

    # CharacterizationManager
    # ------------------------------------------------------------------------------------------------------------------
    def check_metrics_networks(self, metrics_to_compute):
        for metric in metrics_to_compute:
            if metric not in available_network_metrics:
                m = (f"""{self.file_name} {metric} is not among available ones. Please select only available metrics to compute with NetworkX graph or edge_list format. 
                Available graph network metrics are: {str(available_network_metrics)}.""")
                self.lm.printl(m)
                raise ValueError(m)

    def check_node_metrics(self, metrics_to_compute):
        for metric in metrics_to_compute:
            if metric not in available_node_metrics:
                m = (f"""{self.file_name} {metric} is not among available ones. Please select only available metrics: 
                Available node metrics are: {str(available_node_metrics)}.""")
                self.lm.printl(m)
                raise ValueError(m)

    # def check_metrics_info_networks(self, metrics_to_compute):
    #     for metric in metrics_to_compute:
    #         if metric not in available_NF_metrics_info_to_compute:
    #             m = (f"""{self.file_name} Please select only available metrics to compute on info_edge_list.
    #             Available info network metrics are: {str(available_NF_metrics_info_to_compute)}.""")
    #             self.lm.printl(m)
    #             raise ValueError(m)