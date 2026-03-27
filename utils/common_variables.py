import seaborn as sns
import matplotlib.pyplot as plt

#UK dataset
# dtype = {'id': str, 'userId': str, 'replyId': str, 'retweetId': str, 'replyUserId': str, 'retweetUserId':str,
#          'id_str': str, 'user.id_str': str, 'in_reply_to_status_id_str': str, 'in_reply_to_user_id_str': str,
#          'retweeted_status.id_str': str, 'retweeted_status.user.id_str': str, "userId1": str, "userId2": str,
#          'location': str, 'botometer.is_bot': 'boolean', 'botometer.is_bot_english': 'boolean',
#          'botometer.skipped': 'boolean', 'isBot': 'boolean', 'isBotEnglish': 'boolean', 'botSkipped': 'boolean',
#          'reply': str, 'retweet': str, 'mention': str, 'hashtag': str, 'domainURL': str, 'source': str, 'target': str,
#          'actor': str, 'node': str, 'nodeId': str}

# IORussia dataset
dtype = {'postid': str, 'accountid': str, 'id': str, 'userId': str, 'reposted_accountid': str, 
         'reposted_postid': str, 'in_reply_to_accountid': str, 'in_reply_to_postid': str, 'post_text': str,
        'application_name': str, 'post_language': str, 'post_time': str, 'account_profile_description': str, 
         'account_creation_date': str, 'reply': str, 'retweet': str, 'mention': str, 'hashtag': str, 'domainURL': str, 'source': str, 'target': str,
         'actor': str, 'node': str, 'nodeId': str}

level = {
            "InputManager": -1,
            "SelectionUserManager": 0,
            "SimilarityFunctionManager": 1,
            "FilterGraphManager": 2,
            "NetworkManager": 3,
            "CommunityDetectionManager": 4,
            "CharacterizationManager": 5,
            "OverlappingCommunityManager": 6
        }

available_type_filter = ['top_co_action_original', 'top_co_action_merge_original', 'top_co_action', 'top_co_action_merge', 'most_active_users', 'top_tweeters', 'top_retweeters']

available_co_action = {"co-retweet": ['overlapping', "overlapping_coefficient", "tfidf_cosine_similarity"],
                       "co-reply": ['overlapping', "overlapping_coefficient","tfidf_cosine_similarity"],
                       "co-url-domain": ['overlapping', "overlapping_coefficient","tfidf_cosine_similarity"],
                       "co-mention": ['overlapping', "overlapping_coefficient","tfidf_cosine_similarity"],
                       "co-hashtag": ['overlapping', "overlapping_coefficient","tfidf_cosine_similarity"]}



sparse_computation_function = ["tfidf_cosine_similarity"]

dense_computation_function = ["overlapping",
                              "overlapping_coefficient",
                              "tfidf_cosine_similarity"]

available_type_merge = ["sum", "average"]

# the tuple of edge_list has the following order:
# - userId1, userId2,
# - weight (similarity measure)
# - nAction (number of co-actions contributing to the edge. for "overlapping" is the same of weight
# - twCount (it is present only in the merged edge list of all the windows. In how many time windows the edge appears?)
# - alpha (it is present only after Backbone filtering step, it represents the importance of the edge)
# , "alpha": 5
NODE1_VAR = "userId1"
NODE2_VAR = "userId2"
W_VAR = 'w_'
NA_VAR = "nAction"
TW_VAR = "twCount"
tuple_index = {NODE1_VAR: 0, NODE2_VAR: 1, W_VAR: 2, NA_VAR: 3, TW_VAR: 4}


available_filter_graph = ['low_std', 'mean', 'high_std', 'th', 'median', 'filter_merge_action', 'merge_filter_action', "backbone", "node_topEdge"]

one_layer_algorithm = ["louvain", "infomap"]
multi_layer_algorithm = ["gclique_percolation", 'ginfomap', 'glouvain', 'abacus', 'multidimensional_label_propagation',
                         'flat_ec', 'flat_nw', 'flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain',
                         'flat_ec_infomap', 'flat_nw_infomap', 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']
flatten_algorithm = ['flat_ec', 'flat_nw', 'flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain',
                     'flat_ec_infomap', 'flat_nw_infomap', 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']
custom_flatten_algorithm = ['flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain',
                            'flat_ec_infomap', 'flat_nw_infomap', 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']
multi_temporal_multi_layer_algorithm = []
required_algorithm_parameters = {'louvain': [('resolution', 'increasing the resolution parameter will typically result in more communities')],
                                 'infomap': [],
                        'gclique_percolation': [('k', 'minimum number of layers'), ('m', 'minimum number of actors in a clique')],
                        'glouvain': [('omega', 'inter-layer weight parameter in the generalized louvain method. omega=0 is like performing Louvain on each single layer'),
                                     ('gamma', 'increasing the resolution parameter will typically result in more communities')],
                        'abacus': [('min_actors', 'minimum number of actors'), ('min_layers', 'minimum number of layers')],
                        'ginfomap': [('interlayer_weight', 'inter-layer weight parameter')],
                        'flat_ec_louvain': [('resolution', 'increasing the resolution parameter will typically result in more communities')],
                        'flat_nw_louvain': [('resolution', 'increasing the resolution parameter will typically result in more communities')],
                        'flat_weighted_sum_louvain': [('resolution', 'increasing the resolution parameter will typically result in more communities')],
                        'flat_weighted_average_louvain': [('resolution', 'increasing the resolution parameter will typically result in more communities')],
                        'flat_and_weighted_sum_louvain': [('resolution', 'increasing the resolution parameter will typically result in more communities')],
                        'flat_ec_infomap': [], 'flat_nw_infomap': [], 'flat_weighted_sum_infomap': [], 'flat_weighted_average_infomap': [], 'flat_and_weighted_sum_infomap': []
                                 }

# available_graph_network_metrics = ['weight_statistics', 'NF_nNodes', 'NF_nEdges', 'node_topEdge_trend']
# available_edge_list_network_metrics = ['nNodes', 'nEdges', 'weight_statistics', 'node_topEdge_trend', 'assortativity', 'degree_centrality', 'degree_distribution' 'betweenness_centrality', 'closeness_centrality', 'shortest_path_lengths', 'eccentricity']
available_network_metrics = ['nNodes', 'nEdges', 'weight_statistics', 'connected_components', 'node_topEdge_trend', 'nAction_distribution', 'assortativity', 'degree_centrality', 'degree_distribution' 'betweenness_centrality', 'closeness_centrality', 'shortest_path_lengths', 'eccentricity']
require_network_construction_metrics = ['connected_components', 'assortativity', 'degree_centrality', 'degree_distribution' 'betweenness_centrality', 'closeness_centrality', 'shortest_path_lengths', 'eccentricity']

available_overlapping_metrics = ['absolute', 'intersect_x', 'intersect_y', 'minimum', 'jaccard', 'harmonicMean']

available_node_metrics = ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality",
                        "local_clustering_coefficient", "page_rank"]

comparison_type = ["coverage.actors", "coverage.edges",
                    "jaccard.actors", "jaccard.edges",
                    "jeffrey.degree", "pearson.degree"]
camparison_map = {
    "coverage.actors": "Coverage Actors",
    "coverage.edges": "Coverage Edges",
    "jaccard.actors": "Jaccard Actors",
    "jaccard.edges": "Jaccard Edges",
    "jeffrey.degree": "Jeffrey Degree",
    "pearson.degree": "Pearson Degree"
}

# color_dict = {"co-retweet": "#e41a1c", "co-reply": "#377eb8", "co-url-domain": "#4daf4a", "co-mention": "#984ea3", "co-hashtag": "#ff7f00"}
co_action_column = {"co-retweet": "retweet", "co-reply": "reply", "co-url-domain": "domainUrl", "co-mention": "mention", "co-hashtag": "hashtag"}

co_action_column_print = {"co-retweet": "RTW", "co-reply": "RPL", "co-url-domain": "URL", "co-mention": "MEN", "co-hashtag": "HST"}
co_action_column_print2 = {"retweet": "RTW", "reply": "RPL", "URL": "URL", "mention": "MEN", "hashtag": "HST"}
co_action_column_print3 = {"co-retweet": "RTW", "co-reply": "RPL", "co-URL": "URL", "co-mention": "MEN", "co-hashtag": "HST"}
multimodal_print = {'multimodal': "MUL", "flat_nw_louvain": "UNFL (nw)", "flat_ec_louvain": "UNFL (ec)", "flat_weighted_sum_louvain": "UNFL (sum)",
                    "flat_weighted_average_louvain": "UNFL (avg)", "flat_and_weighted_sum_louvain": "UNFL (and sum)",
                    
                    "flat_nw_infomap": "UNFL (nw)", "flat_ec_infomap": "UNFL (ec)", "flat_weighted_sum_infomap": "UNFL (sum)",
                    "flat_weighted_average_infomap": "UNFL (avg)", "flat_and_weighted_sum_infomap": "UNFL (and sum)"}

co_action_map = {"co-retweet": "co-retweet", "co-reply": "co-reply", "co-url-domain": "co-URL", "co-mention": "co-mention", "co-hashtag": "co-hashtag"}
action_map = {"co-retweet": "retweet", "co-reply": "reply", "co-url-domain": "URL", "co-mention": "mention", "co-hashtag": "hashtag"}
action_map_inverse = {"retweet": "co-retweet", "reply": "co-reply", "URL": "co-url-domain", "mention": "co-mention", "hashtag": "co-hashtag"}
action_map_inverse_print = {"retweet": "co-retweet", "reply": "co-reply", "URL": "co-URL", "mention": "co-mention", "hashtag": "co-hashtag"}
url_map = {'co-url-domain': 'co-URL'}

# abbreviation
co_action_abbreviation_map = {"co-retweet": "rt", "co-reply": "rp", "co-url-domain": "ud", "co-hashtag": "h", "co-mention": "m"}
similarity_function_map = {"overlapping": "o", "overlapping_coefficient": "oc", "tfidf_cosine_similarity": "tics"}
filter_map = {'low_std': "ls", 'mean': "m", 'high_std': "hs", 'th': "th", 'median': 'md', 'filter_merge_action': "fma",
              'merge_filter_action': 'mfa', "backbone": "b", "node_topEdge": "nte"}
algorithm_map = {'louvain': 'l', "clique_percolation": "cp", 'infomap': "im", 'glouvain': "gl", 'abacus': "a",
                 'multidimensional_label_propagation': "mlp", 'flat_ec': "fec", 'flat_nw': "fnw",
                 'flat_ec_louvain': 'fecl', 'flat_nw_louvain': 'fnwl', 'flat_weighted_sum_louvain': 'fwsl',
                 'flat_weighted_average_louvain': 'fwal', 'flat_and_weighted_sum_louvain': 'fawsl',
                 'flat_ec_infomap': 'feci', 'flat_nw_infomap': 'fnwi', 'flat_weighted_sum_infomap': 'fwsi',
                 'flat_weighted_average_infomap': 'fwai', 'flat_and_weighted_sum_infomap': 'fawsi', 'ginfomap': 'gi'}


parameters_map = {'resolution': "res", "min_actors": "a", "min_layers": "l", "omega": "o", 
                  "gamma": "g", "interlayer_weight": "iw", "k": "k", "m": "m"}

# visualization parameters
dpi = 300
heatmap_color = "viridis"
pastel_palette = sns.color_palette("pastel")
palette = {'lost': pastel_palette[3],  # common: blue, gained: green, lost: red
            'common': pastel_palette[0],
           'gained': pastel_palette[2]}

color_dict = {"co-retweet": pastel_palette[0],
              "co-reply": pastel_palette[1],
              "co-url-domain": pastel_palette[2],
              "co-mention": pastel_palette[3],
              "co-hashtag": pastel_palette[5]}
color_dict2 = {"RTW": pastel_palette[0],
              "RPL": pastel_palette[1],
              "URL": pastel_palette[2],
              "MEN": pastel_palette[3],
              "HST": pastel_palette[5]}



