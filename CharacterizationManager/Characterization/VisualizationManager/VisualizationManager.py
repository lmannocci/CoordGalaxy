from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *

import uunet.multinet as ml
import os
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import numpy as np
import math
absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")

class VisualizationManager:
    def __init__(self, list_ca, dict_ca_filter, icm, dm, type_algorithm, cda):
        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.icm = icm
        self.dm = dm
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))
        self.type_algorithm = type_algorithm
        self.cda = cda

    def __find_grid_dimensions(self, nLayer):
        # Start with x and y close to sqrt(N)
        x = math.floor(math.sqrt(nLayer))
        y = math.ceil(math.sqrt(nLayer))

        # Ensure x * y >= N
        while x * y < nLayer:
            if y <= x:
                y += 1
            else:
                x += 1

        return x, y

    def __delete_edges(self, MG, com_df):

        vertices_df = pd.DataFrame(ml.vertices(MG))

        clustered_actor_df = com_df[['actor', 'layer']].copy()
        clustered_actor_df['actor'] = clustered_actor_df['actor'].astype(str)

        # drop the coordinated users, I select only the not clustered users, which I will delete frome the ML subsequently
        concat_df = pd.concat([clustered_actor_df, vertices_df]).reset_index(drop=True).drop_duplicates(
            subset=['actor', 'layer'], keep=False)
        df_dict = concat_df.to_dict(orient='list')
        ml.delete_vertices(MG, df_dict)

        return MG

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    def visualize_multiplex_network(self):
        self.lm.printl(f"{file_name}. visualize_multiplex_network start.")

        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
        # TODO temporal multiplex network not implemented. There is only one file and temporal directory does not exist
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        # read multiplex network
        MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)
        com = self.ch.load_object(self.dm.path_coms + 'coms.p')
        nLayer = len(self.list_ca)

        layout = ml.layout_multiforce(MG)
        nRows, nCols = self.__find_grid_dimensions(nLayer)
        ml.plot(MG, com=com, vertex_labels=[], layout=layout, grid=[nRows, nCols], vertex_size=[4], format='png', file=f'{self.dm.path_community_visualization}{net_filename_no_ext}.png')
        self.lm.printl(f"{file_name}. visualize_multiplex_network completed.")

    def delete_edges_visualize_multiplex_network(self):
        self.lm.printl(f"{file_name}. visualize_multiplex_network start.")

        com_df_files = [pos_csv for pos_csv in os.listdir(self.dm.path_user_dataframe) if pos_csv.endswith('.csv')]
        com_df_filename = com_df_files[0]
        com_df = self.ch.read_dataframe(self.dm.path_user_dataframe + com_df_filename, dtype=dtype)

        if len(com_df['cid'].unique()) < 10:
            graph_files = [pos_csv for pos_csv in os.listdir(self.dm.path_multi_graph) if pos_csv.endswith('.txt')]
            # TODO temporal multiplex network not implemented. There is only one file and temporal directory does not exist
            net_filename = graph_files[0]
            net_filename_no_ext = net_filename.split('.')[0]
            # read multiplex network
            MG = self.ch.read_multiplex_network(self.dm.path_multi_graph + net_filename)

            com_files = [pos_csv for pos_csv in os.listdir(self.dm.path_coms) if pos_csv.endswith('.p')]

            com_filename = com_files[0]

            com = self.ch.load_object(self.dm.path_coms + com_filename)

            MG = self.__delete_edges(MG, com_df)

            layout = ml.layout_multiforce(MG)
            nLayer = len(self.list_ca)

            nRows, nCols = self.__find_grid_dimensions(nLayer)

            ml.plot(MG, com=com, vertex_labels=[], layout=layout, grid=[nRows, nCols], vertex_size=[4], format='png',
                    file=f'{self.dm.path_community_visualization}{net_filename_no_ext}.png')
        else:
            self.lm.printl(f"{file_name}. The number of communities is too high to be visualized.")
        self.lm.printl(f"{file_name}. visualize_multiplex_network completed.")

    def delete_small_communities_single_layer(self, th_size):
        self.lm.printl(f"{file_name}. delete_small_communities started.")
        type_ca = self.list_ca[0].get_co_action()
        graph_files = [pos_csv for pos_csv in os.listdir(self.dm.dict_path_ca[type_ca]['path_filter_graph']) if
                       pos_csv.endswith('.p')]
        net_filename = graph_files[0]
        net_filename_no_ext = net_filename.split('.')[0]
        G = self.ch.load_object(self.dm.path_community_graph + net_filename)

        df = self.ch.read_dataframe(self.dm.path_user_dataframe + "com_df.csv", dtype=dtype)
        agg_df = df.groupby(['group']).size().reset_index(name='nUsers')

        left_com_df = pd.DataFrame(agg_df[agg_df['nUsers'] >= th_size]['group'])
        filter_df = pd.merge(df, left_com_df, on='group', how='inner')
        self.lm.printl(f"{file_name}. Number of communities: {len(left_com_df['group'].unique())}")

        nodes_to_keep = set(filter_df['userId'].values)
        all_nodes = set(G.nodes)
        nodes_to_remove = all_nodes - nodes_to_keep
        self.lm.printl(f"{file_name}. Number of nodes to be removed: {len(nodes_to_remove)}")

        G.remove_nodes_from(nodes_to_remove)

        self.ch.save_dataframe(df, self.dm.path_user_dataframe + f"th_size_{str(th_size)}_com_df.csv")

        self.ch.save_object(G, self.dm.path_community_graph + f"th_size_{str(th_size)}_{net_filename}")

        # save the graph in the gephi format
        self.cm.from_graph_to_gephi(G, self.dm.path_community_gephi_graph + f"th_size_{str(th_size)}_{net_filename_no_ext}.gexf")

        self.lm.printl(f"{file_name}. delete_small_communities completed.")