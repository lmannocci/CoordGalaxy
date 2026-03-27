from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
from Objects.CDAlgorithm.CDAlgorithm import *
import networkx as nx
import os

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class CommunityDetectionManager:
    def __init__(self, dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda: CDAlgorithm):
        """
            Given a dataframe of pasts published by several users, it gets posts by the top user_fraction users.
            :param type_time_window: [str] The type of time window can be: ATW (Adjacent Time Window), OTW (Overlapping Time Window),
            ANY (no time window). The ATW exploits only tw_str, since tw_slide_interval_str is equal to tw_str.
            :param tw_str: [str] Length of the window, e.g., 1d, 1h, 30s.
            :param tw_slide_interval_str: [str] Size of the slide of the window. How much the window scrolls each time.
            :return: [str] Return the path of the directory according to the parameters. ATW is based only on tw_str parameter,
            while OTW both on tw_str and tw_slide_interval_str. Instead, ANY implies no window at all, so it is free parameter.
        """
        self.lm = LogManager('main')
        self.ch = Checkpoint()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.cda = cda

        self.icm = IntegrityConstraintManager(file_name)
        self.dm = DirectoryManager(file_name, dataset_name, results=results, user_fraction=self.user_fraction,
                                   type_filter=self.type_filter, tw=tw, list_ca=list_ca, dict_ca_filter=dict_ca_filter, cda=cda)

        # check if for each co_action in the list, it is passed the corresponding threshold in the dictionary of the threshold.
        self.icm.check_co_action(list_ca, dict_ca_filter)
        # self.icm.check_dict_ca_filter(dict_ca_filter)

        self.icm.check_type_algorithm(tw, list_ca, cda.get_algorithm_name())

        self.type_algorithm = self.dm.get_type_algorithm()
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))

        self.cm = ConversionManager()

    def __from_multiplex_to_single_layer_communities_format(self, df):
        self.lm.printl(f"{file_name}. __from_multiplex_to_single_layer_communities_format started.")
        df = df[['actor', 'cid']]
        df = df.rename(columns={'actor': 'userId', 'cid': 'group'})
        df = df.drop_duplicates(keep='first')

        # The group id are not given, as usual done by Louvain algorithm in order from the
        # community with the highest number of users to the lowest. So we need to assign the group id manually, sorting
        # the groups by the number of users in each group and assigning the lowest ID to the group with the highest
        # number of users.

        # Count the number of users in each group
        group_counts = df['group'].value_counts().sort_values(ascending=False)
        # Create a mapping from group to ID, assigning the lowest ID to the group with fewer users
        group_mapping = {group: idx for idx, group in enumerate(group_counts.index)}
        # Map the 'group' column in the original dataframe to the new group IDs
        df['group'] = df['group'].map(group_mapping)

        communities = []
        for group in df['group'].unique():
            communities.append(df[df['group'] == group]['userId'].tolist())
        self.lm.printl(f"{file_name}. __from_multiplex_to_single_layer_communities_format completed.")
        return communities, df

    def __from_communities_to_user_dataframe(self, communities):
        self.lm.printl(f"{file_name}. __from_communities_to_user_dataframe started.")
        unfolded_communities = {"userId": [], "group": []}
        for ind_com, com in enumerate(communities):
            for node in com:
                unfolded_communities["userId"].append(node)
                unfolded_communities["group"].append(ind_com)

        df = pd.DataFrame(unfolded_communities)
        self.lm.printl(f"{file_name}. __from_communities_to_user_dataframe completed.")

        return df

    def __set_node_attribute_communities(self, G, communities):
        self.lm.printl(f"{file_name}. __set_node_attribute_communities started.")
        coms_dictionary = {}
        for ind_com, com in enumerate(communities):
            for node in com:
                temp_attribute = {}
                temp_attribute['group'] = ind_com
                coms_dictionary[node] = temp_attribute

        nx.set_node_attributes(G, coms_dictionary)
        self.lm.printl(f"{file_name}. __set_node_attribute_communities completed.")
        return G

    def compute_community_detection(self):
        self.lm.printl(f"{file_name}. compute_community_detection algorithm: {str(self.cda)} started.")

        if self.type_algorithm == "one-layer":
            type_ca = self.list_ca[0].get_co_action()
            if self.tw.get_type_output_network() == "merged":
                graph_files = [pos_csv for pos_csv in os.listdir(self.dm.dict_path_ca[type_ca]['path_filter_graph']) if pos_csv.endswith('.p')]
                net_filename = graph_files[0]
                net_filename_no_ext = net_filename.split('.')[0]
                # read single network
                G = self.ch.load_object(self.dm.dict_path_ca[type_ca]['path_filter_graph'] + net_filename)

                coms = self.cda.compute_communities(G)
                communities = coms.communities
                # communities example
                # [[id1, id2], [id5, id6, id7], ...]
                self.ch.save_object(coms, self.dm.path_coms + net_filename)

                # save the community in a dataframe format, (userId, group)
                com_df = self.__from_communities_to_user_dataframe(communities)
                self.ch.save_dataframe(com_df, self.dm.path_user_dataframe + "com_df.csv")

                self.__set_node_attribute_communities(G, communities)
                # save the graph with the information of the community detection
                self.ch.save_object(G, self.dm.path_community_graph + net_filename)

                # save the graph in the gephi format
                self.cm.from_graph_to_gephi(G, self.dm.path_community_gephi_graph + net_filename_no_ext + '.gexf')

            elif self.tw.get_type_output_network() == "temporal":
                pass
        elif self.type_algorithm == "multi-layer":
            if self.tw.get_type_output_network() == "merged":
                graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
                net_filename = graph_files[0]
                net_filename_no_ext = net_filename.split('.')[0]

                MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

                # This snippet is here, because G is required for 'flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum',
                # while it could be moved below for 'flat_ec', 'flat_nw', which do not require G, since they perform the
                # flattening of the multiplex network internally. In any case, I need the flattened network to save it in
                # the gephi format.
                if self.cda.get_algorithm_name() in flatten_algorithm:
                    G = self.cda.flatten_multiplex_network(MG)
                    self.ch.save_object(G, self.dm.path_community_graph + net_filename_no_ext + '.p')

                if self.cda.get_algorithm_name() in custom_flatten_algorithm:
                    coms = self.cda.compute_communities(G)
                    communities = coms.communities
                    com_df = self.__from_communities_to_user_dataframe(communities)
                else:
                    coms = self.cda.compute_communities(MG)
                    com_df = self.cm.to_df(coms)
                    # these two algorithms return the communities in the multiplex format, even if they are performed on
                    # the flattened network. We need to convert them to the single layer format.
                    if self.cda.get_algorithm_name() in list(set(flatten_algorithm) - set(custom_flatten_algorithm)): #  ['flat_ec', 'flat_nw']
                        communities, com_df = self.__from_multiplex_to_single_layer_communities_format(com_df)

                self.ch.save_object(coms, self.dm.path_coms + 'coms.p')
                self.ch.save_dataframe(com_df, self.dm.path_user_dataframe + "com_df.csv")

                # in these case, we are as in the single layer setting
                if self.cda.get_algorithm_name() in flatten_algorithm:
                    self.__set_node_attribute_communities(G, communities)
                    # save the graph with the information of the community detection
                    self.ch.save_object(G, self.dm.path_community_graph + net_filename_no_ext + '.p')
                    # save the graph in the gephi format
                    self.cm.from_graph_to_gephi(G, self.dm.path_community_gephi_graph + net_filename_no_ext + '.gexf')

            elif self.tw.get_type_output_network() == "merged":
                pass
        self.lm.printl(f"{file_name}. compute_community_detection algorithm: {str(self.cda)} completed.")

