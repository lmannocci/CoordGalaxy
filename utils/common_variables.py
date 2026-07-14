import matplotlib.pyplot as plt
from dataclasses import dataclass, field

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

# Default dataframe dtypes used by Checkpoint reads. Example: userId, objectId, and contentType are read as strings
# so identifiers are not accidentally converted to numbers.
# dtype = {'id': str, 'userId': str, 'replyId': str, 'retweetId': str, 'replyUserId': str, 'retweetUserId':str,
#          'id_str': str, 'user.id_str': str, 'in_reply_to_status_id_str': str, 'in_reply_to_user_id_str': str,
#          'retweeted_status.id_str': str, 'retweeted_status.user.id_str': str, "userId1": str, "userId2": str,
#          'location': str, 'botometer.is_bot': 'boolean', 'botometer.is_bot_english': 'boolean',
#          'botometer.skipped': 'boolean', 'isBot': 'boolean', 'isBotEnglish': 'boolean', 'botSkipped': 'boolean',
#          'reply': str, 'retweet': str, 'mention': str, 'hashtag': str, 'domainURL': str, 'source': str, 'target': str,
#          'actor': str, 'node': str, 'nodeId': str, 'text': str, 'content': str}

dtype = {'postid': str, 'accountid': str, 'id': str, 'userId': str, 'reposted_accountid': str, 
         'reposted_postid': str, 'in_reply_to_accountid': str, 'in_reply_to_postid': str, 'post_text': str,
        'application_name': str, 'post_language': str, 'post_time': str, 'account_profile_description': str, 
         'account_creation_date': str, 'reply': str, 'retweet': str, 'mention': str, 'hashtag': str, 'domainURL': str, 'source': str, 'target': str,
         'actor': str, 'node': str, 'nodeId': str, 'text': str, 'content': str, 'objectId': str, 'contentType': str}

# Pipeline stage levels used by DirectoryManager to build the right directory tree for each manager.
# Example: SimilarityFunctionManager has level 1, while CharacterizationManager has level 5.
level = {
            "InputManager": -1,
            "SelectionUserManager": 0,
            "SimilarityFunctionManager": 1,
            "FilterGraphManager": 2,
            "NetworkManager": 3,
            "CommunityDetectionManager": 4,
            "CharacterizationManager": 5,
            "LatexTableManager": 5,
            "CommunityComparisonManager": 6
        }

# User-selection strategies accepted by SelectionUserManager. Example: top_co_action_merge.
available_type_filter = ['top_co_action_original', 'top_co_action_merge_original', 'top_co_action', 'top_co_action_merge', 'most_active_users', 'top_tweeters', 'top_retweeters']

# Standard object column used in all normalized co-action CSV files. Example: objectId stores a hashtag, URL domain, or comment id.
DEFAULT_OBJECT_COLUMN = "objectId"
# Similarity functions available for object-set co-actions. Example: co-hashtag can use tfidf_cosine_similarity.
DEFAULT_SIMILARITY_FUNCTIONS = ('overlapping', "overlapping_coefficient", "tfidf_cosine_similarity")
# Similarity functions available for text-embedding co-actions. Example: co-commentText can use average_cosine_similarity.
TEXT_SIMILARITY_FUNCTIONS = ('average_cosine_similarity', 'overlapping_coefficient', 'overlapping')


@dataclass(frozen=True)
class CoActionSpec:
    """
        Define one framework co-action and all names derived from it.
        :param co_action_id: [str] Canonical framework id used in configs and code, e.g., co-hashtag.
        :param layer_name: [str] File stem and multiplex layer name. It must avoid dashes and underscores for uunet.
        :param display_name: [str] Human-readable name used in reports and plots.
        :param short_label: [str] Compact label used in plots.
        :param abbreviation: [str] Compact token used in filenames.
        :param cross_platform_id: [str] Platform-neutral interaction id for future cross-platform analysis.
        :param similarity_functions: [tuple[str, ...]] Similarity functions supported by this co-action.
        :param object_column: [str] Standard object column in co-action CSV files.
        :param has_embeddings: [bool] Whether the co-action has an aligned embedding artifact.
        :param aliases: [tuple[str, ...]] Additional input names accepted for this co-action.
        :param platform_display_names: [dict[str, str]] Optional platform-specific display labels.
    """
    co_action_id: str
    layer_name: str
    display_name: str
    short_label: str
    abbreviation: str
    cross_platform_id: str
    similarity_functions: tuple[str, ...] = DEFAULT_SIMILARITY_FUNCTIONS
    object_column: str = DEFAULT_OBJECT_COLUMN
    has_embeddings: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)
    platform_display_names: dict[str, str] = field(default_factory=dict)


# Single source of truth for framework co-actions. Example: co-hashtag uses layer/file name hashtag and plot label HST.
CO_ACTION_SPECS = (
    CoActionSpec(
        co_action_id="co-retweet",
        layer_name="retweet",
        display_name="Retweet",
        short_label="RT",
        abbreviation="rt",
        cross_platform_id="co-reshare",
        platform_display_names={"twitter": "Retweet"},
        aliases=("retweet",),
    ),
    CoActionSpec(
        co_action_id="co-share",
        layer_name="share",
        display_name="Share",
        short_label="SHR",
        abbreviation="sh",
        cross_platform_id="co-reshare",
        platform_display_names={"facebook": "Share"},
        aliases=("share",),
    ),
    CoActionSpec(
        co_action_id="co-reply",
        layer_name="reply",
        display_name="Reply",
        short_label="RPL",
        abbreviation="rp",
        cross_platform_id="co-response",
        platform_display_names={"twitter": "Reply"},
        aliases=("reply",),
    ),
    CoActionSpec(
        co_action_id="co-comment",
        layer_name="comment",
        display_name="Comment",
        short_label="CMT",
        abbreviation="c",
        cross_platform_id="co-response",
        platform_display_names={"facebook": "Comment", "reddit": "Comment", "moltbook": "Comment"},
        aliases=("comment",),
    ),
    CoActionSpec(
        co_action_id="co-url-domain",
        layer_name="URL",
        display_name="URL domain",
        short_label="URL",
        abbreviation="url",
        cross_platform_id="co-url",
        aliases=("co-url", "URL", "url", "domainURL"),
    ),
    CoActionSpec(
        co_action_id="co-mention",
        layer_name="mention",
        display_name="Mention",
        short_label="MEN",
        abbreviation="m",
        cross_platform_id="co-mention",
        aliases=("mention", "MNT", "MEN"),
    ),
    CoActionSpec(
        co_action_id="co-hashtag",
        layer_name="hashtag",
        display_name="Hashtag",
        short_label="HST",
        abbreviation="h",
        cross_platform_id="co-hashtag",
        aliases=("hashtag", "HTG", "HST"),
    ),
    CoActionSpec(
        co_action_id="co-commentText",
        layer_name="commentText",
        display_name="Comment text",
        short_label="CTXT",
        abbreviation="ct",
        cross_platform_id="co-response-text",
        similarity_functions=TEXT_SIMILARITY_FUNCTIONS,
        has_embeddings=True,
        aliases=("commentText",),
    ),
    CoActionSpec(
        co_action_id="co-commentURL",
        layer_name="commentURL",
        display_name="Comment URL",
        short_label="CURL",
        abbreviation="cu",
        cross_platform_id="co-response-url",
        aliases=("commentURL",),
    ),
    CoActionSpec(
        co_action_id="co-postText",
        layer_name="postText",
        display_name="Post text",
        short_label="PTXT",
        abbreviation="pt",
        cross_platform_id="co-post-text",
        similarity_functions=TEXT_SIMILARITY_FUNCTIONS,
        has_embeddings=True,
        aliases=("postText",),
    ),
    CoActionSpec(
        co_action_id="co-postURL",
        layer_name="postURL",
        display_name="Post URL",
        short_label="PURL",
        abbreviation="pu",
        cross_platform_id="co-post-url",
        aliases=("postURL",),
    ),
)

# Canonical co-action registry keyed by framework id. Example: CO_ACTION_REGISTRY["co-comment"].
CO_ACTION_REGISTRY = {spec.co_action_id: spec for spec in CO_ACTION_SPECS}
# Alias-to-canonical lookup for configs and plotted layer labels. Example: HST and hashtag map to co-hashtag.
CO_ACTION_ALIASES = {
    alias: spec.co_action_id
    for spec in CO_ACTION_SPECS
    for alias in (spec.co_action_id, spec.layer_name, *spec.aliases)
}


def normalize_co_action_id(co_action_id: str) -> str:
    """
        Return the canonical framework co-action id for a configured or displayed co-action name.
        :param co_action_id: [str] Co-action id, layer name, or accepted alias.
        :return: [str] Canonical framework co-action id.
    """
    return CO_ACTION_ALIASES.get(co_action_id, co_action_id)


def get_co_action_spec(co_action_id: str) -> CoActionSpec:
    """
        Return the co-action specification for a framework co-action id or alias.
        :param co_action_id: [str] Co-action id, layer name, or accepted alias.
        :return: [CoActionSpec] Co-action metadata.
    """
    canonical_id = normalize_co_action_id(co_action_id)
    return CO_ACTION_REGISTRY[canonical_id]


def get_co_action_layer_name(co_action_id: str) -> str:
    """
        Return the file stem and uunet-safe layer name for a co-action.
        :param co_action_id: [str] Co-action id or alias.
        :return: [str] Layer/file name.
    """
    return get_co_action_spec(co_action_id).layer_name


def get_co_action_display_name(co_action_id: str, platform: str | None = None) -> str:
    """
        Return a human-readable display name for a co-action.
        :param co_action_id: [str] Co-action id or alias.
        :param platform: [str | None] Optional platform name used to prefer platform-specific labels.
        :return: [str] Display name.
    """
    spec = get_co_action_spec(co_action_id)
    if platform is not None and platform in spec.platform_display_names:
        return spec.platform_display_names[platform]
    return spec.display_name


def get_co_action_short_label(co_action_id: str) -> str:
    """
        Return a compact plot label for a co-action.
        :param co_action_id: [str] Co-action id or alias.
        :return: [str] Short label.
    """
    return get_co_action_spec(co_action_id).short_label


def get_co_action_cross_platform_id(co_action_id: str) -> str:
    """
        Return the platform-neutral interaction id for a co-action.
        :param co_action_id: [str] Co-action id or alias.
        :return: [str] Cross-platform interaction id.
    """
    return get_co_action_spec(co_action_id).cross_platform_id


# Supported similarity functions by canonical co-action id. Example: available_co_action["co-postText"].
available_co_action = {
    co_action_id: list(spec.similarity_functions)
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}

# Co-actions that have an aligned embedding artifact next to their CSV. Example: co-commentText has commentText.npy.
co_action_embeddings = [
    co_action_id
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
    if spec.has_embeddings
]

# Similarity functions that support sparse matrix computation. Example: tfidf_cosine_similarity.
sparse_computation_function = ["tfidf_cosine_similarity"]

# Similarity functions that support pairwise dense computation. Example: overlapping_coefficient.
dense_computation_function = ["overlapping",
                              "overlapping_coefficient",
                              "tfidf_cosine_similarity"]

# Similarity functions whose sparse/dense choice is not relevant. Example: average_cosine_similarity uses embeddings.
irrelevant_sparse_computation_function = ["average_cosine_similarity"]

# Merge operators for temporal edge lists. Example: average averages edge weights across time windows.
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

# Graph filtering strategies accepted by FilterGraphManager. Example: th applies a fixed threshold.
available_filter_graph = ['low_std', 'mean', 'high_std', 'th', 'median', 'filter_merge_action', 'merge_filter_action', "backbone", "node_topEdge"]

# Single-layer community-detection algorithms. Example: louvain.
one_layer_algorithm = ["louvain", "infomap"]
# Multiplex community-detection and flattening algorithms. Example: glouvain and flat_weighted_sum_louvain.
multi_layer_algorithm = ["gclique_percolation", 'ginfomap', 'glouvain', 'abacus', 'multidimensional_label_propagation',
                         'flat_ec', 'flat_nw', 'flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain',
                         'flat_ec_infomap', 'flat_nw_infomap', 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']
# Algorithms that flatten multiplex networks into one layer. Example: flat_nw_louvain.
flatten_algorithm = ['flat_ec', 'flat_nw', 'flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain',
                     'flat_ec_infomap', 'flat_nw_infomap', 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']
# Flattening algorithms implemented by this framework rather than directly by the external library.
custom_flatten_algorithm = ['flat_ec_louvain', 'flat_nw_louvain', 'flat_weighted_sum_louvain', 'flat_weighted_average_louvain', 'flat_and_weighted_sum_louvain',
                            'flat_ec_infomap', 'flat_nw_infomap', 'flat_weighted_sum_infomap', 'flat_weighted_average_infomap', 'flat_and_weighted_sum_infomap']
# Multiplex temporal community algorithms. Empty until a temporal multiplex algorithm is added.
multi_temporal_multi_layer_algorithm = []
# Required parameters by community algorithm. Example: louvain requires resolution.
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
# Network metrics available in characterization. Example: nEdges and weight_statistics.
available_network_metrics = [
    'nNodes', 'nEdges', 'weight_statistics', 'connected_components', 'nConnectedComponents',
    'sizeLargestComponent', 'density', 'directed', 'clusteringCoefficient', 'averagePathLength', 'diameter',
    'node_topEdge_trend', 'nAction_distribution', 'assortativity', 'degree_centrality', 'degree_distribution',
    'betweenness_centrality', 'closeness_centrality', 'shortest_path_lengths', 'eccentricity'
]
# Metrics that require constructing a NetworkX graph. Example: connected_components.
require_network_construction_metrics = [
    'connected_components', 'nConnectedComponents', 'sizeLargestComponent', 'density', 'directed',
    'clusteringCoefficient', 'averagePathLength', 'diameter', 'assortativity', 'degree_centrality',
    'degree_distribution', 'betweenness_centrality', 'closeness_centrality', 'shortest_path_lengths', 'eccentricity'
]

# Overlap metrics for comparing actor/community sets. Example: jaccard.
available_overlapping_metrics = ['absolute', 'intersect_x', 'intersect_y', 'minimum', 'jaccard', 'harmonicMean']

# Node-level centrality metrics. Example: page_rank.
available_node_metrics = ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality",
                        "local_clustering_coefficient", "page_rank"]

# Cross-network comparison metrics. Example: coverage.actors.
comparison_type = ["coverage.actors", "coverage.edges",
                    "jaccard.actors", "jaccard.edges",
                    "jeffrey.degree", "pearson.degree"]
# Display names for comparison metrics. Example: coverage.actors becomes Coverage Actors.
camparison_map = {
    "coverage.actors": "Coverage Actors",
    "coverage.edges": "Coverage Edges",
    "jaccard.actors": "Jaccard Actors",
    "jaccard.edges": "Jaccard Edges",
    "jeffrey.degree": "Jeffrey Degree",
    "pearson.degree": "Pearson Degree"
}

# Object column by co-action id. Example: co-hashtag -> objectId.
co_action_column = {
    co_action_id: spec.object_column
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}

# Short plot label by canonical co-action id. Example: co-hashtag -> HST.
co_action_column_print = {
    co_action_id: spec.short_label
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}
# Short plot label by layer/file name. Example: hashtag -> HST.
co_action_column_print2 = {
    spec.layer_name: spec.short_label
    for spec in CO_ACTION_REGISTRY.values()
}
# Short plot label by either canonical id or layer/file name. Example: co-hashtag and hashtag both map to HST.
co_action_column_print3 = co_action_column_print | co_action_column_print2

# Plot labels for multimodal and flattened algorithms. Example: flat_nw_louvain -> UNFL (nw).
multimodal_print = {'multimodal': "MUL", "flat_nw_louvain": "UNFL (nw)", "flat_ec_louvain": "UNFL (ec)", "flat_weighted_sum_louvain": "UNFL (sum)",
                    "flat_weighted_average_louvain": "UNFL (avg)", "flat_and_weighted_sum_louvain": "UNFL (and sum)",                    
                    "flat_nw_infomap": "UNFL (nw)", "flat_ec_infomap": "UNFL (ec)", "flat_weighted_sum_infomap": "UNFL (sum)",
                    "flat_weighted_average_infomap": "UNFL (avg)", "flat_and_weighted_sum_infomap": "UNFL (and sum)"}

# Layer/file name by canonical co-action id. Example: co-commentText -> commentText.
co_action_map = {
    co_action_id: spec.layer_name
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}
# Backward-compatible alias for co_action_map.
action_map = co_action_map.copy()
# Canonical co-action id by layer/file name. Example: commentText -> co-commentText.
action_map_inverse = {
    spec.layer_name: co_action_id
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}
# Backward-compatible alias for action_map_inverse.
action_map_inverse_print = action_map_inverse.copy()


# Abbreviation by canonical co-action id for compact result paths. Example: co-commentURL -> cu.
co_action_abbreviation_map = {
    co_action_id: spec.abbreviation
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}

# Abbreviation by similarity function for compact result paths. Example: overlapping_coefficient -> oc.
similarity_function_map = {"overlapping": "o", "overlapping_coefficient": "oc", "tfidf_cosine_similarity": "tics", "average_cosine_similarity": "acs"}

# Abbreviation by graph filter for compact result paths. Example: merge_filter_action -> mfa.
filter_map = {'low_std': "ls", 'mean': "m", 'high_std': "hs", 'th': "th", 'median': 'md', 'filter_merge_action': "fma",
              'merge_filter_action': 'mfa', "backbone": "b", "node_topEdge": "nte"}
# Abbreviation by community algorithm for compact result paths. Example: louvain -> l.
algorithm_map = {'louvain': 'l', "clique_percolation": "cp", 'infomap': "im", 'glouvain': "gl", 'abacus': "a",
                 'multidimensional_label_propagation': "mlp", 'flat_ec': "fec", 'flat_nw': "fnw",
                 'flat_ec_louvain': 'fecl', 'flat_nw_louvain': 'fnwl', 'flat_weighted_sum_louvain': 'fwsl',
                 'flat_weighted_average_louvain': 'fwal', 'flat_and_weighted_sum_louvain': 'fawsl',
                 'flat_ec_infomap': 'feci', 'flat_nw_infomap': 'fnwi', 'flat_weighted_sum_infomap': 'fwsi',
                 'flat_weighted_average_infomap': 'fwai', 'flat_and_weighted_sum_infomap': 'fawsi', 'ginfomap': 'gi'}

# Abbreviation by community algorithm parameter for compact result paths. Example: resolution -> res.
parameters_map = {'resolution': "res", "min_actors": "a", "min_layers": "l", "omega": "o", 
                  "gamma": "g", "interlayer_weight": "iw", "k": "k", "m": "m"}

# Default plot resolution.
dpi = 300
# Default colormap name for heatmaps.
heatmap_color = "viridis"
# Pastel colors used for co-action and overlap plots. Uses seaborn when installed, otherwise matplotlib.
if sns is not None:
    pastel_palette = sns.color_palette("pastel")
else:
    pastel_palette = plt.get_cmap("Pastel1").colors
# Colors for overlap-state plots. Example: gained/common/lost stacked bars.
palette = {'lost': pastel_palette[3],  # common: blue, gained: green, lost: red
            'common': pastel_palette[0],
           'gained': pastel_palette[2]}


# Color by canonical co-action id. Example: color_dict["co-comment"].
color_dict = {
    co_action_id: pastel_palette[index % len(pastel_palette)]
    for index, co_action_id in enumerate(CO_ACTION_REGISTRY.keys())
}
# Same co-action colors keyed by short plot label. Example: color_dict2["CMT"] == color_dict["co-comment"].
color_dict2 = {
    spec.short_label: color_dict[co_action_id]
    for co_action_id, spec in CO_ACTION_REGISTRY.items()
}

# Stable LaTeX color by top multiplex community id for Moltbook table/plot consistency.
# Example: community 18 is rendered as \textcolor{red}{\Large $\bullet$}.
top_community_latex_color_map = {
    "18": "red",
    "60": "blue",
    "19": "orange",
    "39": "black",
    "67": "cyan",
    "3": "purple",
    "62": "magenta",
    "5": "brown",
    "31": "teal",
    "16": "violet",
    "17": "green",
    "24": "gray",
    "12": "olive",
    "28": "pink",
    "59": "lime",
    "9": "yellow",
    "27": "red!70!black",
    "6": "blue!70!black",
    "64": "orange!70!black",
}
