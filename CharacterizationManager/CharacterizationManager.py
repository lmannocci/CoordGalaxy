from CharacterizationManager.Characterization.NetworkMeasure.NetworkMeasure import *
from CharacterizationManager.Characterization.NodeMeasure.NodeMeasure import NodeMeasure
from CharacterizationManager.Characterization.VisualizationManager.VisualizationManager import *
from CharacterizationManager.Characterization.MultiplexCommunitiesEvaluation.MultiplexCommunitiesEvaluation import *
from CharacterizationManager.Characterization.SingleLayerCommunitiesEvaluation.SingleLayerCommuntiesEvaluation import *
from CharacterizationManager.Characterization.NodeMeasure.NodeMeasure import *
from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *

import uunet.multinet as ml
import os
import numpy as np

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")
data_path = os.path.join(absolute_path, f"..{os.sep}data{os.sep}")

class CharacterizationManager:
    def __init__(self, dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter, cda=None):
        """
        """
        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter
        self.cda = cda

        self.icm = IntegrityConstraintManager(file_name)
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path, results=results, user_fraction=self.user_fraction,
                                   type_filter=self.type_filter, tw=tw, list_ca=list_ca,
                                   dict_ca_filter=dict_ca_filter, cda=cda)

        self.type_algorithm = self.dm.get_type_algorithm()

        # check if for each co_action in the list, it is passed the corresponding threshold in the dictionary of the threshold.
        self.icm.check_co_action(list_ca, dict_ca_filter)

        if cda is not None:
            self.icm.check_type_algorithm(tw, list_ca, cda.get_algorithm_name())

        self.type_algorithm = self.dm.get_type_algorithm()
        self.list_ca_str = '_'.join(list(self.dm.dict_path_ca.keys()))

        self.nm = NetworkMeasure(self.list_ca, self.dict_ca_filter, self.icm, self.dm, self.type_algorithm)
        self.nom = NodeMeasure(self.list_ca, self.dict_ca_filter, self.icm, self.dm, self.type_algorithm, self.cda)
        self.vm = VisualizationManager(self.list_ca, self.dict_ca_filter, self.icm, self.dm, self.type_algorithm, self.cda)
        self.mce = MultiplexCommunitiesEvaluation(self.dataset_name, self.user_fraction, self.type_filter, 
                                                  self.list_ca, self.dict_ca_filter, 
                                                  self.icm, self.dm, self.type_algorithm, self.cda)
        
        self.sle = SingleLayerCommunitiesEvaluation(self.dataset_name, self.user_fraction, self.type_filter, 
                                                    self.list_ca, self.dict_ca_filter, 
                                                    self.icm, self.dm, self.type_algorithm, self.cda)

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------

    # NetworkMeasure
    # ------------------------------------------------------------------------------------------------------------------
    def compute_threshold_statistics(self, min_th, max_th, step, filter_par_type):
        self.lm.printl(f"{file_name}. compute_threshold_statistics start.")
        self.nm.compute_threshold_statistics(min_th, max_th, step, filter_par_type)
        self.lm.printl(f"{file_name}. compute_threshold_statistics completed.")

    def plot_threshold_statistics(self, filter_par_type, step):
        self.nm.plot_threshold_overlapping(filter_par_type, step)
        self.nm.plot_nodes_edges_threshold(filter_par_type)

    def select_threshold_statistics(self, min_th, max_th, step, absolute_th_mode, filter_par_type, target_type):
        self.nm.select_threshold_statistics(min_th, max_th, step, absolute_th_mode, filter_par_type, target_type)

    def compute_metrics_networks(self, metrics_to_compute):
        self.lm.printl(f"{file_name}. compute_metrics_networks start.")
        self.nm.compute_metrics_network(metrics_to_compute)
        self.lm.printl(f"{file_name}. compute_metrics_networks completed.")

    def edge_weight_temporal_mean_std(self):
        self.lm.printl(f"{file_name}. edge_weight_temporal_mean_std start.")
        self.nm.edge_weight_temporal_mean_std()
        self.lm.printl(f"{file_name}. edge_weight_temporal_mean_std completed.")

    # Multiplex characterization before CDA
    def get_ML_layer_comparison(self):
        self.lm.printl(f"{file_name}. get_ML_layer_comparison start.")
        self.nm.get_ML_layer_comparison()
        self.lm.printl(f"{file_name}. get_ML_layer_comparison completed.")

    def plot_ML_layer_comparison(self):
        self.lm.printl(f"{file_name}. plot_ML_layer_comparison start.")
        self.nm.plot_ML_layer_comparison()
        self.lm.printl(f"{file_name}. plot_ML_layer_comparison completed.")

    def get_ML_summary(self):
        self.lm.printl(f"{file_name}. get_ML_summary start.")
        self.nm.get_ML_summary()
        self.lm.printl(f"{file_name}. get_ML_summary completed.")

    def compute_node_metrics(self, metrics, merge_existing=False):
        self.lm.printl(f"{file_name}. compute_node_metrics start.")
        self.nom.compute_node_metrics(metrics, merge_existing)
        self.lm.printl(f"{file_name}. compute_node_metrics completed.")

    # MultiplexCommunitiesEvaluation and SingleLayerCommunitiesEvaluation (after CDA)
    # ------------------------------------------------------------------------------------------------------------------
    def compute_statistics_communities(self):
        self.lm.printl(f"{file_name}. compute_statistics_communities start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            self.sle.compute_statistics_communities()
        else:
            self.mce.compute_statistics_communities()
        self.lm.printl(f"{file_name}. compute_statistics_communities completed.")

    def compute_info_communities(self):
        self.lm.printl(f"{file_name}. compute_info_communities start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            self.lm.printl(f"{file_name}. compute_info_communities implemented only for multiplex community discovery algorithm.")
        else:
            self.mce.compute_info_communities()
        self.lm.printl(f"{file_name}. compute_info_communities completed.")

    def compute_metrics_communities(self, community_size_th):
        self.lm.printl(f"{file_name}. compute_metrics_communities start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            self.sle.compute_metrics_communities(community_size_th)
        else:
            self.lm.printl(f"{file_name}. compute_metrics_communities implemented only for single layer community discovery algorithm.")
        self.lm.printl(f"{file_name}. compute_metrics_communities completed.")

    def compute_metrics_nodes_communities(self, metrics=None, th_size=None, restrict_neighbors=True, merge_existing=False):
        self.lm.printl(f"{file_name}. compute_metrics_nodes_communities start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            self.sle.compute_metrics_node_communities(metrics, th_size, restrict_neighbors, merge_existing)
        else:
            self.mce.compute_metrics_node_communities(metrics, th_size, restrict_neighbors, merge_existing)
        self.lm.printl(f"{file_name}. compute_metrics_nodes_communities completed.")

    def validate_communities(self):
        self.lm.printl(f"{file_name}. validate_communities start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            self.sle.validate_communities()
        else:
            self.mce.validate_communities()
        self.lm.printl(f"{file_name}. validate_communities completed.")

    def compute_coordination_communities(self,  community_size_th=None, community_label="group", weight_label="w_"):
        self.lm.printl(f"{file_name}. compute_coordination_communities start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            if self.cda.get_algorithm_name() == 'flat_nw_infomap' or self.cda.get_algorithm_name() == 'flat_nw_louvain':
                self.lm.printl(f"{file_name}. compute_coordination_communities not applicable for infomap and louvain flatten algorithms (no weights).")
            else:
                self.sle.compute_coordination_communities(community_size_th, community_label, weight_label)
        else:
            self.mce.compute_coordination_communities(community_size_th, community_label, weight_label)
        self.lm.printl(f"{file_name}. compute_coordination_communities completed.")

    # VisualizationManager
    # ------------------------------------------------------------------------------------------------------------------
    def visualize_multiplex_network(self):
        self.lm.printl(f"{file_name}. visualize_multiplex_network start.")
        self.vm.visualize_multiplex_network()
        self.lm.printl(f"{file_name}. visualize_multiplex_network completed.")

    def delete_edges_visualize_multiplex_network(self):
        self.lm.printl(f"{file_name}. delete_edges_visualize_multiplex_network start.")
        if self.type_algorithm == 'one-layer' or self.cda.get_algorithm_name() in flatten_algorithm:
            self.lm.printl(f"{file_name}. delete_edges_visualize_multiplex_network implemented only for multiplex community discovery algorithm.")
        else:
            self.vm.delete_edges_visualize_multiplex_network()
        self.lm.printl(f"{file_name}. delete_edges_visualize_multiplex_network completed.")

    def delete_small_communities_single_layer(self, th_size):
        self.lm.printl(f"{file_name}. delete_small_communities_single_layer started.")
        self.vm.delete_small_communities_single_layer(th_size)
        self.lm.printl(f"{file_name}. delete_small_communities_single_layer completed.")


    # GENERAL PURPOSE OBJECTS FUNCTIONS
    def get_directory_manager(self):
        return self.dm

    def get_checkpoint(self):
        return self.ch

    def get_type_algorithm(self):
        return self.type_algorithm

    def get_list_ca(self):
        return self.list_ca

    def get_dict_ca_filter(self):
        return self.dict_ca_filter

    def get_cda(self):
        return self.cda
