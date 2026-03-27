from Objects.CoAction.CoAction import *
from Objects.TimeWindow.TimeWindow import *
from Objects.Filter.Filter import *
from Objects.CDAlgorithm.CDAlgorithm import *
from utils.mainMethods import *

# InputManager data
dataset_name = "uk"
startDate = get_formatted_date("2019-11-12 00:00:00")
endDate = get_formatted_date("2019-12-12 23:59:00")



# RETRIEVE TWEETS INFO FROM ELASTIC
# sourceList = select_sources_info_tweet_ES()
# elastic_info = {"username_index": "nizzoli", "index_name": "extended_ukelections19_tweets", "type_query": "dateSources",
#                 "startDate": startDate, "endDate": endDate, "sourceList": sourceList,
#                 "filename": "text_uk_tweets19.json"}

# RETRIEVE USERS INFO FROM ELASTIC
sourceList = select_sources_info_user_ES()
elastic_info = {"username_index": "nizzoli", "index_name": "ukelections19_users", "type_query": "allSources",
                "sourceList": sourceList, "filename": "1_uk_final_users.csv"}

# InputManager data

# Extract URL
# these domains are known domains, which must not be unshortened (since this operation takes long time)
known_url = ['twitter.com', 'cards.twitter.com', 'youtube.com', 'instagram.com', 'facebook.com',
             'open.spotify.com', 'google.com', 'reddit.com', 'play.google.com', 'bing.com', 'google.co.uk',
             'paper.li', 'theguardian.com', 'bbc.co.uk', 'independent.co.uk', 'mirror.co.uk', 'telegraph.co.uk',
             'thesun.co.uk', 'mobile.twitter.com', 'metro.co.uk', 'dailymail.co.uk', 'express.co.uk',
             'thetimes.co.uk', 'bbc.com', 'news.sky.com']

# these domains areare excluded from the final dataset, since they are very shared domain
excludeDomainList = ['twitter.com', 'cards.twitter.com', 'youtube.com', 'instagram.com', 'facebook.com',
                     'open.spotify.com', 'google.com', 'reddit.com', 'play.google.com', 'bing.com', 'google.co.uk']

# these hashtags areare excluded from the final dataset, since they have been used for the crawling
# Before filtering #rows: 5827266, # distincthashtags: 63425
# After filtering #rows: 3502507, # distincthashtags: 63412
excludeHashtagList = ['GE2019', 'GeneralElection19', 'GeneralElection19', 'GeneralElection2019', 'VoteLabour',
                      'VoteLabour2019', 'ForTheMany', 'ForTheManyNotTheFew', 'ChangeIsComing', 'RealChange',
                      'VoteConservative', 'VoteConservative2019', 'BackBoris', 'GetBrexitDone']

# these mentions areare excluded from the final dataset, since they have been used for the crawling
excludeMentionDict = {'jeremycorbyn': '117777690', 'UKLabour': '14291684', '3131144855': 'BorisJohnson',
                      'Conservatives': '14281853'}
excludeMentionList = list(excludeMentionDict.values())

filter_dataset = {
    'co-retweet': False,
    'co-reply': False,
    'co-url-domain': True,
    'co-mention': True,
    'co-hashtag': True
}


# SelectionUserManager
user_fraction = 0.05
type_filter = "top_co_action_merge"  # 'top_co_action', 'top_co_action_merge', 'most_active_users', 'top_tweeters', 'top_retweeters'

# TimeWindow
type_output_network = "merged"  # merged, temporal
type_time_window = "OTW"
tw_str = "6h"
tw_slide_interval_str = "5h"  # not considered in case of ATW
type_merge = "average"  # mandatory only for type_output_network = "merged"
tw = TimeWindow(type_output_network, type_time_window, tw_str, tw_slide_interval_str, type_merge=type_merge)

# CoAction
similarity_function = "tfidf_cosine_similarity"  # overlapping_coefficient
co_action_list = ["co-retweet", "co-reply", "co-mention", "co-hashtag", "co-url-domain"]

# co_action = "co-mention"
# ca = CoAction(co_action, similarity_function)

# FilterGraphManager
# Filter 1
# filter_instance = Filter("filter_merge_action", 3, None)
# Filter 2.1
# filter_instance = Filter("node_topEdge", 1500, Filter("filter_merge_action", 3, None))


# No filter
# list_ca = [CoAction("co-retweet", "overlapping_coefficient"), CoAction("co-reply", "overlapping_coefficient"),
#            CoAction("co-url-domain", "overlapping_coefficient"), CoAction("co-mention", "overlapping_coefficient"),
#            CoAction("co-hashtag", "overlapping_coefficient")]
list_ca = [CoAction("co-retweet", "tfidf_cosine_similarity"), CoAction("co-reply", "tfidf_cosine_similarity"),
           CoAction("co-url-domain", "tfidf_cosine_similarity"), CoAction("co-mention", "tfidf_cosine_similarity"),
           CoAction("co-hashtag", "tfidf_cosine_similarity")]

# CharacterizationManager: No-Filter
# dict_ca_filter = {"co-retweet": None,
#                   'co-reply': None,
#                   'co-url-domain': None,
#                   'co-mention': None,
#                   'co-hashtag': None}

# Filter 1
# dict_ca_filter = {"co-retweet": Filter("merge_filter_action", 26, None),
#                   'co-reply': Filter("merge_filter_action", 4, None),
#                   'co-url-domain': Filter("merge_filter_action", 3, None),
#                   'co-mention': Filter("merge_filter_action", 48, None),
#                   'co-hashtag': Filter("merge_filter_action", 14, None)}

# Filter 2.1 threshold to have 1'500'000 edges
# dict_ca_filter = {"co-retweet": Filter("th", 0.14, Filter("merge_filter_action", 26, None)),
#                   'co-reply': Filter("th", 0.43, Filter("merge_filter_action", 4, None)),
#                   'co-url-domain': Filter("th", 0.88, Filter("merge_filter_action", 3, None)),
#                   'co-mention': Filter("th", 0.16, Filter("merge_filter_action", 48, None)),
#                   'co-hashtag': Filter("th", 0.25, Filter("merge_filter_action", 14, None))}
# Filter 2.2 median (very similar to 2.1)
dict_ca_filter = {"co-retweet": Filter("median", 0.126, Filter("merge_filter_action", 26, None)),
                  'co-reply': Filter("median", 0.451, Filter("merge_filter_action", 4, None)),
                  'co-url-domain': Filter("median", 0.791, Filter("merge_filter_action", 3, None)),
                  'co-mention': Filter("median", 0.111, Filter("merge_filter_action", 48, None)),
                  'co-hashtag': Filter("median", 0.246, Filter("merge_filter_action", 14, None))}

# Filter 2.3
# dict_ca_filter = {"co-retweet": Filter("node_topEdge", 1500, Filter("threshold_action", 3, None)),
#                   'co-reply': Filter("node_topEdge", 1500, Filter("threshold_action", 3, None)),
#                   'co-url-domain': Filter("node_topEdge", 1500, Filter("threshold_action", 3, None)),
#                   'co-mention': Filter("node_topEdge", 1500, Filter("threshold_action", 3, None)),
#                   'co-hashtag': Filter("node_topEdge", 1500, Filter("threshold_action", 3, None))}


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
