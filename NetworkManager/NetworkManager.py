from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.ConversionManager.ConversionManager import *
from utils.Checkpoint.Checkpoint import *
from utils.common_variables import *
import networkx as nx
import uunet.multinet as ml
import os
import pandas as pd

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class NetworkManager:
    def __init__(self, dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter):
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

        self.icm = IntegrityConstraintManager(file_name)
        self.dm = DirectoryManager(file_name, dataset_name, results=results, user_fraction=self.user_fraction,
                                   type_filter=self.type_filter, tw=tw, list_ca=list_ca, dict_ca_filter=dict_ca_filter)
        self.cm = ConversionManager()

        # check if for each co_action in the list, it is passed the corresponding threshold in the dictionary of the threshold.
        self.icm.check_co_action(list_ca, dict_ca_filter)
        # self.icm.check_dict_ca_filter(dict_ca_filter)

        self.type_algorithm = self.dm.type_algorithm
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    def create_weighted_graph(self):
        """
            Convert filtered edge list in networkx graph.
        """
        self.lm.printl(f"{file_name}. create_weighted_graph start.")

        for type_ca, dict_path in self.dm.dict_path_ca.items():
            edge_list_files = [pos_csv for pos_csv in os.listdir(dict_path["path_filter_edge_list"]) if pos_csv.endswith('.p')]
            for elf in edge_list_files:
                self.lm.printl(f"{file_name}. Creating graph for co-action {type_ca}, time window: {elf}, filter: {repr(self.dict_ca_filter[type_ca])}.")
                edge_list = self.ch.load_object(dict_path["path_filter_edge_list"] + elf)
                # G = nx.Graph()
                # G.add_weighted_edges_from(edge_list, weight='weight')

                G = self.cm.from_edge_list_to_graph(edge_list)

                self.lm.printl(f"{file_name}. create_weighted_graph, {elf}.")
                self.ch.save_object(G, dict_path["path_filter_graph"] + elf)

        self.lm.printl(f"{file_name}. create_weighted_graph completed.")

    def create_weighted_multiplex_network(self, save_edge_list_df=False):
        """
            Creates multiplex network from the single co-action graph. It is not implemented for considering multiple time window.
            For this reason it expects only to work in case of "merged" type of output.
        :return: [PyMLNetwork] Multiplex network of multiple co-actions.
        """
        self.lm.printl(f"{file_name}. create_weighted_multiplex_network start for co-actions {self.list_ca_str}")
        MG = ml.empty()

        for type_ca, dict_path in self.dm.dict_path_ca.items():
            layer = action_map[type_ca]
            self.lm.printl(f"START LAYER: {layer}.")
            graph_files = [pos_csv for pos_csv in os.listdir(dict_path["path_filter_graph"]) if pos_csv.endswith('.p')]
            if len(graph_files) == 0:
                m = f"{file_name}. Please, run create_weighted_graph() before calling this method. NetworkX Graph are required to create a multiplex network."
                self.lm.printl(m)
                raise Exception(m)

            elf = graph_files[0]
            G = self.ch.load_object(dict_path["path_filter_graph"] + elf)

            MG = self.cm.add_layer_multiplex_network(MG, G, layer)
            if save_edge_list_df==True:
                edge_list_df = nx.to_pandas_edgelist(G)
                edge_list_df['layer'] = layer
                self.ch.update_dataframe(edge_list_df, self.dm.path_multi_edge_list_df + "edge_list_df.csv", dtype=dtype)

        self.ch.save_multiplex_network(MG, self.dm.path_multi_graph + "multiplex_graph.txt")
        self.lm.printl(self.cm.to_df(ml.attributes(MG, target="edge")))
        for type_ca, dict_path in self.dm.dict_path_ca.items():
            layer = action_map[type_ca]
            edge_layer_df = self.cm.to_df(ml.edges(MG, layers1=[layer]))
            self.lm.printl(edge_layer_df.shape)

        self.lm.printl(f"{file_name}. create_weighted_multiplex_network completed.")

    def save_gephi_network(self):
        self.lm.printl(f"{file_name}. save_gephi_network start for co-actions {self.list_ca_str}")

        for type_ca, dict_path in self.dm.dict_path_ca.items():
            graph_files = [pos_csv for pos_csv in os.listdir(dict_path["path_filter_graph"]) if pos_csv.endswith('.p')]
            if len(graph_files) == 0:
                m = (f"""{file_name}. Please, run create_weighted_graph() before calling this method. "
                     NetworkX Graph are required to create a Gephi format.""")
                self.lm.printl(m)
                raise Exception(m)

            elf = graph_files[0]
            G = self.ch.load_object(dict_path["path_filter_graph"] + elf)
            net_filename_no_ext = elf.split('.')[0]
            filename = net_filename_no_ext + '.gexf'
            self.cm.from_graph_to_gephi(G, dict_path['path_filter_gephi_graph'] + filename)

        self.lm.printl(f"{file_name}. save_gephi_network completed.")
