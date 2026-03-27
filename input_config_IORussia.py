from Objects.CoAction.CoAction import *
from Objects.TimeWindow.TimeWindow import *
from Objects.Filter.Filter import *
from Objects.CDAlgorithm.CDAlgorithm import *
from utils.mainMethods import *

# InputManager data
dataset_name = "IORussia"

# Extract URL
# these domains are known domains, which must not be unshortened (since this operation takes long time)
known_url = []

# these domains areare excluded from the final dataset, since they are very shared domain
excludeDomainList = []

# these hashtags areare excluded from the final dataset, since they have been used for the crawling
excludeHashtagList = []

# these mentions areare excluded from the final dataset, since they have been used for the crawling
excludeMentionDict = {}
excludeMentionList = list(excludeMentionDict.values())

filter_dataset = {
    'co-retweet': False,
    'co-reply': False,
    'co-url-domain': False,
    'co-mention': False,
    'co-hashtag': False
}

# SelectionUserManager
user_fraction = 1.0
type_filter = "top_co_action_merge"  # 'top_co_action', 'top_co_action_merge', 'most_active_users', 'top_tweeters', 'top_retweeters'

# TimeWindow
type_output_network = "merged"  # merged, temporal
type_time_window = "OTW"
tw_str = "7d"
tw_slide_interval_str = "6d"  # not considered in case of ATW
type_merge = "average"  # mandatory only for type_output_network = "merged"
tw = TimeWindow(type_output_network, type_time_window, tw_str, tw_slide_interval_str, type_merge=type_merge)

# CoAction
similarity_function = "tfidf_cosine_similarity"  # overlapping_coefficient
co_action_list = ["co-reply"]

# co_action = "co-mention"
# ca = CoAction(co_action, similarity_function)

# FilterGraphManager
# Filter 1
# filter_instance = Filter("filter_merge_action", 3, None)
# Filter 2.1
# filter_instance = Filter("node_topEdge", 1500, Filter("filter_merge_action", 3, None))


# No filter
list_ca = [CoAction("co-retweet", "tfidf_cosine_similarity"), CoAction("co-reply", "tfidf_cosine_similarity"),
           CoAction("co-url-domain", "tfidf_cosine_similarity"), CoAction("co-mention", "tfidf_cosine_similarity"),
           CoAction("co-hashtag", "tfidf_cosine_similarity")]

# CharacterizationManager: No-Filter
dict_ca_filter = {"co-retweet": None,
                  'co-reply': None,
                  'co-url-domain': None,
                  'co-mention': None,
                  'co-hashtag': None}

# Filter 1
dict_ca_filter = {"co-retweet": Filter("merge_filter_action", 2, None),
                  'co-reply': Filter("merge_filter_action", 2, None),
                  'co-url-domain': Filter("merge_filter_action", 2, None),
                  'co-mention': Filter("merge_filter_action", 2, None),
                  'co-hashtag': Filter("merge_filter_action", 7, None)}

# Filter 2 median
dict_ca_filter = {"co-retweet": Filter("median", 0.137, Filter("merge_filter_action", 2, None)),
                  'co-reply': Filter("median", 0.584, Filter("merge_filter_action", 2, None)),
                  'co-url-domain': Filter("median", 0.268, Filter("merge_filter_action", 2, None)),
                  'co-mention': Filter("median", 0.117, Filter("merge_filter_action", 2, None)),
                  'co-hashtag': Filter("median", 0.246, Filter("merge_filter_action", 7, None))}

# CharacterizationManager
# metrics_to_compute = ["weight_statistics", "nNodes", "nEdges", "assortativity", "degree_centrality", "betweenness_centrality", "closeness_centrality",
#                       "shortest_path_lengths", "eccentricity",  "degree_distribution"]

# metrics_to_compute = ["weight_statistics", "nNodes", "nEdges", "assortativity", "degree_centrality", "degree_distribution"]
# metrics_to_compute = ['weight_statistics', 'nNodes', 'nEdges', 'node_topEdge_trend']
metrics_to_compute = ['connected_components', 'weight_statistics', 'nNodes', 'nEdges']

# CommunityDetectionAlgorithm
# (3, 4), (3, 5), (5, 4), (5, 5),  (8, 4), (8, 5), (10, 4), (10, 5), (15, 3), (15, 4), (15, 5)  ---> abacus empty results
# parameters_dict = {'clique_percolation': [(3, 4), (4, 3), (3, 3), (4, 2), (5, 2)],
#                    'glouvain': [(0,), (0.02,), (0.04,), (0.06,), (0.08,), (0.1,), (0.12,), (0.14,), (0.2,), (0.4,), (0.6,), (0.8,), (1,)],
#                    'abacus': [(3, 2), (3, 3), (5, 2), (5, 3), (8, 2), (8, 3), (10, 2), (10, 3), (15, 2)]
#                    }

# Characterization Communities
metrics_node_to_compute = ["degree_centrality", "eigenvector_centrality", "page_rank", "local_clustering_coefficient"]
   
parameters_dict = {
    'ginfomap': [(0.15,)],

    'flat_ec_infomap': [None],
    'flat_nw_infomap': [None],
    'flat_weighted_sum_infomap': [None],
    'flat_and_weighted_sum_infomap': [None],

    'glouvain': [(0.1, 1)],

    'flat_ec_louvain': [(1,)],
    'flat_nw_louvain': [(1,)],
    'flat_weighted_sum_louvain': [(1, )],
    'flat_and_weighted_sum_louvain': [(1,)],
}

single_layer_algorithm_dict = {
    'louvain': [(1,)],
    'infomap': [None],
}


# parameters_dict = {
#     'clique_percolation': [(4, 3), (3, 3), (4, 2), (5, 2)],
#     'abacus': [(3, 2), (3, 3), (5, 2), (5, 3), (8, 2), (8, 3), (10, 2), (10, 3), (15, 2)]
# }

# cda = CDAlgorithm("clique_percolation", {"k": 2, 'm': 2})
