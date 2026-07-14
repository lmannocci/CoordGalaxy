# CoordGalaxy

[**CoordGalaxy**](https://github.com/lmannocci/CoordGalaxy) is an extensible framework for coordinated behavior
detection and characterization across online social platforms.

CoordGalaxy standardizes heterogeneous social-platform datasets into common co-action inputs, builds single-layer and
multiplex coordination networks, supports optional user selection and graph filtering, and provides modules for network
characterization, community detection, community comparison, and LaTeX-ready result tables.

![CoordGalaxy metaphor: the galaxy is the social media space, users are stars, and coordinated communities are constellations.](docs/assets/coordgalaxy.png)

## Future Work

CoordGalaxy is designed to support extensions beyond the current single-platform and time-windowed coordination
pipeline. Two planned directions are especially important:

```text
Cross-platform analysis
```

Future versions will support coordinated behavior analysis across multiple social platforms in the same study. This
requires a shared co-action ontology, platform-specific aliases for equivalent behaviors, and result views that can
compare coordination patterns across ecosystems such as Twitter/X, Reddit, Telegram, YouTube, or domain-specific
communities.

```text
Temporal analysis
```

The current framework already builds time-windowed edge lists and merged networks. A richer temporal-analysis layer is
planned to study how coordinated communities appear, persist, split, merge, and disappear over time, and to characterize
the temporal stability of users, co-actions, layers, and communities.

## Adding a New Dataset

Dataset-specific input logic lives in `InputManager/dataset/`. To add a dataset, create a new preprocessing class that extends `DatasetPreprocessing` and register it in `InputManager/dataset/__init__.py`.

Example:

```python
from .base_preprocessing import DatasetPreprocessing


class MyDatasetPreprocessing(DatasetPreprocessing):
    ...
```

Register the class:

```python
from .my_dataset_preprocessing import MyDatasetPreprocessing

DATASET_PREPROCESSORS = {
    "my_dataset": MyDatasetPreprocessing,
}
```

The dataset name used in the config and main file must match the registry key, for example `dataset_name = "my_dataset"`.

### Required Normalization Method

Every dataset class must implement:

```python
def normalize_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
    ...
```

This method receives the raw dataset dataframe and must save a normalized intermediate CSV in `data/<dataset>/temp_data/` using:

```python
self.ch.save_dataframe(df, self.dm.path_temp_data + filename)
```

The normalized dataframe must contain, at minimum:

```text
id,userId,created,contentType
```

If the dataset contains labels, keep them as optional columns such as:

```text
isControl
```

Before returning, call:

```python
df = self.standardize_normalized_data(df)
```

This renames any legacy `type` column to `contentType` and converts original `userId` values to simple framework ids using `data/<dataset>/temp_data/user_id_mapping.json`.

### Optional Normalization Methods

Some datasets have extra files. Implement these only when needed:

```python
def normalize_user_data(self, df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
    ...

def normalize_data_text(self, df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
    ...
```

If they are not implemented, the base class logs that the step is unavailable and returns `None`.

### Required Extraction Methods

Every dataset class must implement these dataset-specific selectors:

```python
def extract_url_data(self, df: pd.DataFrame) -> pd.DataFrame:
    ...

def extract_hashtag_data(self, df: pd.DataFrame) -> pd.DataFrame:
    ...

def extract_mention_data(self, df: pd.DataFrame) -> pd.DataFrame:
    ...

def extract_retweet_data(self, df: pd.DataFrame) -> pd.DataFrame:
    ...

def extract_reply_data(self, df: pd.DataFrame) -> pd.DataFrame:
    ...

def extract_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
    ...
```

These methods should only select and reshape dataset-specific columns. The base class handles the common saving and standardization through methods such as:

```python
extract_url_dataset(...)
extract_hashtag_dataset(...)
extract_mention_dataset(...)
extract_retweet_dataset(...)
extract_reply_dataset(...)
extract_text_dataset(...)
```

Final co-action files are saved in:

```text
data/<dataset>/co_action_data/
```

Each final co-action CSV must have:

```text
id,userId,created,objectId,contentType
```

Optional label columns, such as `isControl`, are preserved when present.

If a dataset does not support a co-action, return:

```python
self.empty_standard_dataframe()
```

or, when the dataset has `isControl` labels:

```python
self.empty_standard_dataframe(include_is_control=True)
```

### Utility Classes

Use shared utilities instead of duplicating general logic:

```text
InputManager/utils/urls_processing.py
InputManager/utils/mentions_hashtags_processing.py
InputManager/utils/id_mapping.py
```

Use `URLPreprocessor` for extracting, parsing, and unshortening URLs. Use `MentionHashtagPreprocessor` for parsing/exploding list-like mention and hashtag values. Use `UserIdMapper` for original-to-simple user id mapping and reverse mapping.

### Information Operations Datasets

Some datasets belong to the same Information Operations collection and share the same input schema. The framework keeps
their shared preprocessing in:

```text
InputManager/dataset/information_operation_preprocessing.py
```

Dataset-specific classes for this collection should inherit `InformationOperationPreprocessing` and only override
methods when the dataset really differs from the shared schema. For example:

```python
class Russia1Preprocessing(InformationOperationPreprocessing):
    ...

class Iran5Preprocessing(InformationOperationPreprocessing):
    ...
```

The current Information Operations datasets are based on the ICWSM paper DOI
https://doi.org/10.1609/icwsm.v19i1.35958 and the Zenodo dataset record
https://zenodo.org/records/14189193. Use stable dataset keys such as `russia1` and `iran5` in configs, main files, data
directories, and `DATASET_PREPROCESSORS`.

### MH Reddit Dataset

The MH Reddit dataset is registered with the dataset key `mh`. Put the raw sample or full dataset at:

```text
data/mh/original/mh.csv
```

The MH preprocessor normalizes Reddit rows into `data/mh/temp_data/comments.csv` and extracts:

```text
data/mh/co_action_data/comment.csv
data/mh/co_action_data/commentText.csv
```

`comment.csv` is the active `co-comment` layer in `configs/mh.py`. `commentText.csv` is extracted for the future text
layer, but `main_mh.py` currently calls text extraction with `build_embeddings=False`, so no `.npy` embedding file is
created during the default preprocessing run.

### Main File and File Layout

A dataset main script should read original source files from:

```text
data/<dataset>/original/
```

For example, Iran5 stores its raw CSV as:

```text
data/iran5/original/iran5.csv
```

The main script should read these files with `read_original_file(...)`, then normalize them into:

```text
data/<dataset>/temp_data/
```

and extract final co-action artifacts into:

```text
data/<dataset>/co_action_data/
```

For text co-actions, save both:

```text
<layer>.csv
<layer>.npy
```

The `.csv` stores the standardized text co-action rows, and the `.npy` stores the embeddings aligned to the CSV row order.

Do not include the dataset name in generated filenames when the dataset directory already provides that context. Prefer names based on the co-action layer, such as:

```text
comment.csv
commentText.csv
commentText.npy
commentURL.csv
postText.csv
postText.npy
postURL.csv
```

## Configuration and Main Scripts

Each dataset can have its own `main_<dataset>.py`, because the pipeline order can be slightly different across datasets.
For example, one dataset may have users files, another may have text embeddings, and another may skip user selection.

The recommended pattern is to keep dataset settings in `configs/<dataset>.py` and load them through one shared
entry point:

```python
from configs import load_config

config = load_config("moltbook")
```

The config file should expose:

```python
def get_config() -> PipelineConfig:
    ...
```

and group user-editable settings into clear blocks:

```text
dataset and input settings
user-selection settings
time-window settings
co-action settings
filter dictionaries
network/community metrics
community-detection algorithms
similarity-computation parameters
```

Even when each dataset has its own main, shared plumbing should not be duplicated. For standard paths and simple CSV
reads, import:

```python
from utils.pipeline_io import DatasetPaths, build_paths, read_dataset_file, read_original_file, read_temp_file
```

These helpers cover:

```text
data/<dataset>/original/
data/<dataset>/
data/<dataset>/temp_data/
data/<dataset>/co_action_data/
```

Keep dataset-specific decisions in `main_<dataset>.py`, such as which raw files to normalize, which extraction methods
to call, and which pipeline steps to skip or reorder. Prefer reading settings as `config.dataset_name`,
`config.list_ca`, and `config.co_action_filters["final"]` instead of using wildcard imports from the config file.

## Adding a New Co-Action

Framework co-actions are defined in `utils/common_variables.py` through `CoActionSpec`. Add a new co-action by appending one entry to `CO_ACTION_SPECS`.

Example:

```python
CoActionSpec(
    co_action_id="co-share",
    layer_name="share",
    display_name="Share",
    short_label="SHR",
    abbreviation="sh",
    cross_platform_id="co-reshare",
    platform_display_names={"facebook": "Share"},
    aliases=("share",),
)
```

Use `co_action_id` as the canonical framework name in configs, such as `CoAction("co-share", "overlapping")`. It should be stable and explicit, usually with the `co-` prefix.

Use `layer_name` as the file stem and multiplex layer name. It must not contain dashes or underscores because the `uunet` multiplex module expects simple layer names. For example, `co-hashtag` uses `hashtag`, and `co-commentText` uses `commentText`.

Use `short_label` only for plots and compact tables. For example, `co-hashtag` has `HST`, while `co-mention` has `MEN`.

Use `cross_platform_id` to group platform-specific co-actions that represent the same behavior. For example, Twitter `co-reply` and Reddit/Facebook `co-comment` both use `co-response`; Twitter `co-retweet` and Facebook `co-share` both use `co-reshare`.

If the co-action uses text embeddings, set:

```python
similarity_functions=TEXT_SIMILARITY_FUNCTIONS,
has_embeddings=True,
```

After adding the spec, the compatibility dictionaries are generated automatically: `available_co_action`, `co_action_map`, `action_map`, `action_map_inverse`, `co_action_column`, `co_action_column_print`, `co_action_abbreviation_map`, `color_dict`, and `color_dict2`.

## User Selection

`SelectionUserManager` is an optional step. Use it only when you want to select a subset of users before computing the co-action networks.

If no user filtering is needed, do not call `apply_user_selection(...)` and keep `user_fraction=None` in the next modules. They will read the original co-action files from:

```text
data/<dataset>/co_action_data/
```

When user selection is applied, the selected co-action files are saved in a sibling directory that records the threshold and selection strategy:

```text
data/<dataset>/co_action_data_th_<user_fraction>_<type_filter>/
```

For example:

```text
data/moltbook/co_action_data_th_0.5_top_co_action_merge/
```

Downstream modules read from this selected directory when it exists for the configured `user_fraction` and `type_filter`. If no user-selection directory exists, they read the original co-action files from `data/<dataset>/co_action_data/`.

Use `analyze_user_selection(...)` to compute and save selection statistics without writing filtered co-action datasets. Use `apply_user_selection(...)` to save the selected co-action datasets.

Recommended interpretation of `user_fraction`:

```text
None -> no user-selection output directory; downstream reads co_action_data/
1.0  -> save all selected users into co_action_data_th_1.0_<type_filter>/
0.5  -> save the selected top 50% users into co_action_data_th_0.5_<type_filter>/
0.05 -> save the selected top 5% users into co_action_data_th_0.05_<type_filter>/
```

`type_filter` controls how the selected users are chosen:

```text
top_co_action                -> select top users separately for each co-action layer
top_co_action_merge          -> select top users per co-action, merge those user sets, then apply the merged set to every layer
top_co_action_original       -> like top_co_action, but URL/mention/hashtag layers use only rows with contentType == original
top_co_action_merge_original -> like top_co_action_merge, but URL/mention/hashtag layers use only original-content rows
most_active_users            -> select users with the most normalized posts, regardless of contentType
top_tweeters                 -> select users with the most original posts
top_retweeters               -> select users with the most retweet co-actions
```

For multiplex analysis, `top_co_action_merge` is usually the most convenient option because every layer is filtered to
the same selected-user set. For example, Iran5 uses `user_fraction=0.05` and `type_filter="top_co_action_merge"` to keep
the top 5% users from the merged co-action selection.

`filter_dataset` tells `SelectionUserManager` whether to read a previously filtered co-action input for each layer:

```python
filter_dataset = {
    "co-url-domain": False,
    "co-mention": False,
    "co-hashtag": False,
    "co-reply": False,
    "co-retweet": False,
}
```

Use `False` for normal files such as `URL.csv`, `mention.csv`, and `retweet.csv`. Use `True` only when you intentionally
created filtered input files such as `URL_filtered.csv` and want user selection to read those instead.

Common `PipelineConfig` parameters:

```text
dataset_name                 -> dataset directory and preprocessor key, e.g., iran5
known_url                    -> domains that URL preprocessing should not unshorten
exclude_domain_list          -> URL objects to remove when optional content filtering is used
exclude_hashtag_list         -> hashtag objects to remove when optional content filtering is used
exclude_mention_dict         -> mention labels/ids to remove when optional content filtering is used
co_action_list               -> canonical co-actions to process in user selection and similarity
list_ca                      -> CoAction objects, including similarity function for each layer
co_action_filters["no_filter"] -> unfiltered network path configuration
co_action_filters["n_action"]  -> first filtering stage, commonly merge_filter_action
co_action_filters["final"]     -> final filtering stage used for graph/community/network outputs
similarity_parallelize_window  -> number of parallel workers/windows for similarity computation
```

## Similarity Computation Settings

`SimilarityFunctionManager` controls how co-action files are converted into edge lists. It is configured with a
`TimeWindow` object and one `CoAction` object.

### TimeWindow Object

`TimeWindow` defines how the input co-action rows are split in time and how the output networks are saved.

Example:

```python
tw = TimeWindow(
    type_output_network="merged",
    type_time_window="ATW",
    tw_str="1d",
    tw_slide_interval_str="1d",
    type_merge="average",
)
```

`type_output_network` controls whether the framework saves one edge list per time window or one merged edge list:

```text
temporal -> save one edge list for each time window
merged   -> merge temporal edge lists into one final edge list
```

`type_time_window` controls how windows are built:

```text
ATW -> adjacent time windows; the slide interval is forced to tw_str
OTW -> overlapping time windows; use tw_slide_interval_str as the slide
ANY -> one window covering the whole dataset
```

`tw_str` is the window length, such as `1d`, `12h`, `30m`, or `10s`. `tw_slide_interval_str` is used only for `OTW`.

`type_merge` is used only when `type_output_network="merged"`:

```text
sum     -> sum edge weights across windows
average -> average edge weights across windows where the edge appears
```

### CoAction Object

`CoAction` identifies which co-action layer must be computed and which similarity function must be used.

Example:

```python
ca = CoAction("co-commentText", "average_cosine_similarity")
```

The first parameter is the canonical co-action id, or one of its aliases defined in `utils/common_variables.py`.
Examples are `co-comment`, `co-commentText`, `co-commentURL`, `co-postText`, and `co-postURL`.

The second parameter is the similarity function. Common choices are:

```text
tfidf_cosine_similarity     -> object-set co-actions such as URLs, hashtags, mentions, replies
average_cosine_similarity   -> embedding/text co-actions such as co-commentText and co-postText
overlapping                 -> count-based overlap
overlapping_coefficient     -> normalized overlap coefficient
```

### SimilarityFunctionManager Parameters

Recommended defaults are:

```python
SimilarityFunctionManager(
    dataset_name,
    user_fraction,
    type_filter,
    tw,
    ca,
    text_similarity_threshold=0.7,
    sparse_computation=False,
    save_info=False,
    parallelize_window=True,
    parallelize_similarity=False,
    merge_info_edge_list=False,
    text_similarity_chunk_size=None,
)
```

Network merging is controlled by the `TimeWindow` object passed as `tw`, not by `SimilarityFunctionManager` directly.
If `tw.get_type_output_network()` is `temporal`, the framework keeps one edge list per time window under
`edge_list/temporal/`. If it is `merged`, temporal edge lists are merged into one edge list under `edge_list/`.

`text_similarity_threshold` is used only for text-embedding co-actions such as `co-commentText` and `co-postText`. It is the minimum cosine/dot-product similarity required for two text objects to count as similar. Suggested value: `0.7`.

Text-embedding co-actions are computed with chunked matrix multiplication. For each time window, the framework compares chunks of text embeddings against all text embeddings in the window, keeps only cross-user text pairs with similarity greater than or equal to `text_similarity_threshold`, and aggregates those matched text pairs into user-user edges. This avoids the old nested Python loop over every pair of users and every pair of texts.

`text_similarity_chunk_size` controls how many text rows are used in each matrix-multiplication chunk. A chunk is compared against all text rows in the same time window, so all valid text-text combinations are still considered; the parameter only controls how much of the similarity matrix is materialized at once.

Approximate memory used by one chunk similarity matrix:

```text
text_similarity_chunk_size * n_texts_in_largest_window * 4 bytes
```

The multiplier is 4 because the similarity matrix is stored as `float32`. This is per active worker process. If `parallelize_window` is high, several workers can allocate one chunk matrix at the same time.

Suggested starting values:

```text
parallelize_window=1-4   -> text_similarity_chunk_size=50000
parallelize_window=8-16  -> text_similarity_chunk_size=10000
parallelize_window=32+   -> text_similarity_chunk_size=2000 or 5000
```

For a machine with about 1 TB RAM and `parallelize_window=70`, start with:

```python
text_similarity_chunk_size = 2000
```

If memory usage stays low, increase it gradually, for example to `5000` and then `10000`. If memory usage is high or workers are killed by the operating system, decrease it. Suggested default for unknown machines: `None`, which lets the framework choose a conservative memory-bounded chunk size.

`sparse_computation` switches object-based co-actions to a sparse matrix implementation where supported, currently for `tfidf_cosine_similarity`. Suggested value: `False`, because the dense pairwise path is the default and is easier to inspect/debug.

`save_info` saves the object-level evidence behind edges, for example which URLs, hashtags, or text pairs contributed to each user-user edge. Suggested value: `False`, because these files can become very large. Enable it only for diagnostics or small experiments. Temporal info files are saved under `info_edge_list/temporal/`.

`merge_info_edge_list` merges temporal info-edge CSV files into one info-edge CSV when the network output is merged. Suggested value: `False`. The merge runs only when all these conditions are true: `tw.get_type_output_network() == "merged"`, `save_info=True`, and `merge_info_edge_list=True`.

`parallelize_window` parallelizes the computation across time windows. Suggested value: `True`, or an integer process count such as `70` on a large machine, because this usually improves computation time.

`parallelize_similarity` parallelizes pairwise user similarity inside each time window for object-based co-actions. Suggested value: `False`; experiments did not show practical advantages, and the process-management overhead can dominate. Text-embedding co-actions use the chunked matrix path instead.

## FilterGraphManager

`FilterGraphManager` is used after similarity computation to reduce edge lists. It now receives the full list of
co-actions and the corresponding filter dictionary, so all layers are filtered with one call.

Example:

```python
dict_ca_filter2 = {
    "co-comment": Filter("merge_filter_action", 2, None),
    "co-commentText": Filter("merge_filter_action", 2, None),
    "co-commentURL": Filter("merge_filter_action", 2, None),
    "co-postText": Filter("merge_filter_action", 12, None),
    "co-postURL": Filter("merge_filter_action", 5, None),
}

fm = FilterGraphManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter2)
fm.filter_graph()
```

### Filter Object

`Filter` has three parameters:

```python
Filter(type_filter, threshold=None, previous_filter=None)
```

`type_filter` is the filtering strategy:

```text
merge_filter_action -> filter a merged edge list by nAction
filter_merge_action -> filter temporal edge lists by nAction before merging
th                  -> filter by edge weight w_ using a fixed threshold
median              -> compute the median edge weight from the input edge list and filter by it
mean                -> compute the mean edge weight and filter by it
low_std             -> compute mean - std and filter by it
high_std            -> compute mean + std and filter by it
backbone            -> apply the disparity-filter backbone method
node_topEdge        -> keep strongest edges while limiting the number of retained nodes
```

`threshold` is interpreted according to `type_filter`. For `merge_filter_action` and `filter_merge_action`, it is a
minimum `nAction`. For `th`, it is a fixed `w_` threshold between 0 and 1. For `backbone`, it is the alpha cutoff.
For `node_topEdge`, it is the maximum number of nodes. For `median`, `mean`, `low_std`, and `high_std`, it must be
`None` because the threshold is computed from the edge list. Passing a numeric threshold for these data-driven filters
raises an error because it would make the config ambiguous.

For data-driven weight filters (`median`, `mean`, `low_std`, `high_std`), the configured threshold is not the final
value used in output paths. It must be `None` until `DirectoryManager` reads the previous edge-list directory and
computes the real threshold. For example:

```python
Filter("median", None, Filter("merge_filter_action", 2, None))
```

means:

```text
1. read the edge lists produced by merge_filter_action_2/
2. compute the median edge weight from those edge lists
3. update the filter threshold, for example to 0.908
4. save/read downstream outputs from merge_filter_action_2/median_0.908/
```

This also means that later stages such as network creation, community detection, and characterization must use the same
filter chain and must be able to access the previous-filter edge lists. If `merge_filter_action_2/` or its `edge_list/`
files are missing, the pipeline should fail instead of creating a new `median_0.0/` branch. A missing intermediate
directory usually means the required previous pipeline stage has not been run, or the config no longer matches the
already generated results.

`previous_filter` links filters in sequence. For example, first filter by `nAction`, then filter the result by weight:

```python
Filter("th", 0.85, Filter("merge_filter_action", 2, None))
```

The filter chain is also used to build output paths, so downstream modules can read the correct filtered edge lists.

## Community Detection

Community detection is configured in `configs/<dataset>.py` and executed with `CommunityDetectionManager`.

Define algorithm parameters directly as dictionaries, not positional tuples. This keeps the config self-explanatory and
means adding a new community-detection algorithm does not require updating a central `get_algorithm_param` mapping.

Example:

```python
single_layer_algorithm_dict = {
    "louvain": [{"resolution": 1}],
    "infomap": [None],
}

multiplex_algorithm_dict = {
    # "ginfomap": [{"interlayer_weight": 0.15}],
    "glouvain": [{"omega": 0.1, "gamma": 1}],
    # "flat_weighted_sum_louvain": [{"resolution": 1}],
}
```

Build the `CDAlgorithm` object directly from these values:

```python
cda = CDAlgorithm(algorithm) if parameters is None else CDAlgorithm(algorithm, parameters)
```

Single-layer community detection must receive one co-action object and the matching one-entry filter dictionary:

```python
single_layer_ca = [ca_by_name["co-comment"]]
single_layer_filter = {"co-comment": config.co_action_filters["final"]["co-comment"]}

cdm = CommunityDetectionManager(
    config.dataset_name,
    config.user_fraction,
    config.type_filter,
    config.tw,
    single_layer_ca,
    single_layer_filter,
    cda,
)
cdm.compute_community_detection()
```

Multiplex community detection receives all co-actions and the full filter dictionary:

```python
cdm = CommunityDetectionManager(
    config.dataset_name,
    config.user_fraction,
    config.type_filter,
    config.tw,
    config.list_ca,
    config.co_action_filters["final"],
    cda,
)
cdm.compute_community_detection()
```

## CharacterizationManager

`CharacterizationManager` computes descriptive statistics and plots for networks before or after filtering. Pass the
same `list_ca` and `dict_ca_filter` that describe the network version you want to characterize.

No-filter characterization:

```python
chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter)
chm.compute_threshold_statistics(2, 480, 5, "nAction")
chm.plot_threshold_statistics("nAction", 10)
chm.compute_network_metrics(metrics_to_compute)
```

Filtered-network characterization:

```python
chm = CharacterizationManager(dataset_name, user_fraction, type_filter, tw, list_ca, dict_ca_filter2)
chm.compute_threshold_statistics(0.01, 0.99, 0.01, "w_")
chm.plot_threshold_statistics("w_", 0.01)
chm.compute_network_metrics(metrics_to_compute)
```

By default, `compute_network_metrics` updates the existing `network_metrics.csv` output. If the file already contains
some requested metric columns, those metrics are not recomputed; only missing requested metrics are added. This is useful
when you already computed basic metrics and later add metrics such as `density` or `diameter`.

To force recomputation of requested metrics:

```python
chm.compute_network_metrics(metrics_to_compute, recompute_existing=True)
```

To ignore the previous output and overwrite `network_metrics.csv` from scratch:

```python
chm.compute_network_metrics(metrics_to_compute, use_existing_output=False)
```

For a single-layer network, pass a one-element `list_ca` and a dictionary containing that co-action. For a multiplex
network, pass all layers. The manager detects the structure from `list_ca` and routes the computation to the appropriate
single-layer or multiplex characterization logic.

Typical methods:

```text
compute_threshold_statistics -> inspect how node/edge counts change across thresholds
plot_threshold_statistics    -> plot threshold statistics saved by the previous method
select_threshold_statistics  -> select thresholds according to node or edge ranges
compute_network_metrics     -> compute network metrics for the selected layers
compute_network_node_metrics -> compute node-level metrics on selected single-layer networks
compute_community_summary_statistics
                             -> summarize community sizes and multiplex community coverage
compute_multiplex_community_membership_summary
                             -> save multiplex community membership summaries per layer and per community
compute_metrics_communities -> compute graph metrics on each detected single-layer community
compute_metrics_nodes_communities
                             -> compute node metrics inside detected communities
compute_community_edge_weight_statistics
                             -> compute within-community edge-weight statistics
get_ML_layer_comparison      -> compute and/or plot multiplex layer comparisons
```

`get_ML_layer_comparison` accepts a `mode` parameter:

```python
chm.get_ML_layer_comparison(mode="both")     # default: compute CSVs and plot heatmaps
chm.get_ML_layer_comparison(mode="compute")  # only compute comparison CSVs
chm.get_ML_layer_comparison(mode="plot")     # only plot existing comparison CSVs
```

`compute_network_metrics` also replaces the old multiplex summary call. To compute the former `uunet.summary`
layer-level fields without loading `multiplex_graph.txt`, include these metrics:

```python
metrics_to_compute = [
    "nNodes",
    "nEdges",
    "directed",
    "nConnectedComponents",
    "sizeLargestComponent",
    "density",
    "clusteringCoefficient",
    "averagePathLength",
    "diameter",
]
```

`averagePathLength` and `diameter` are computed on the largest connected component of each layer.

Use the no-filter dictionary when you want to characterize original networks:

```python
dict_ca_filter = {"co-comment": None, "co-commentText": None}
```

Use a filtered dictionary when you want to characterize a filtered network:

```python
dict_ca_filter2 = {"co-comment": Filter("merge_filter_action", 2, None)}
```

## Results Directory

The framework writes outputs under one dataset-specific result root:

```text
results/<dataset>/<type_filter>_<user_fraction>/
```

For example:

```text
results/moltbook/top_co_action_merge_None/
results/iran5/top_co_action_merge_0.05/
results/mh/top_co_action_merge_0.01/
```

`<type_filter>_<user_fraction>` identifies the optional user-selection stage. If no user selection is applied, the
fraction is usually `None`. If user selection is applied, the same setting also determines which co-action input
directory is read from `data/<dataset>/co_action_data_th_<user_fraction>_<type_filter>/`.

### Network Result Layout

The network result path is controlled by the `TimeWindow` object:

```text
results/<dataset>/<type_filter>_<user_fraction>/
└── <output_network>_network/
    └── <type_time_window>/
        └── <time_window_instance>/
            ├── info_tw/
            └── <merge_type>/                         # only for merged networks
```

The path segments mean:

```text
<output_network>       merged or temporal
<type_time_window>     ANY, ATW, or OTW
<time_window_instance> tw_7d, tw_1d-tw_slide_interval_12h, ...
<merge_type>           average, sum, or another merge strategy configured in TimeWindow
```

Examples:

```text
results/mh/top_co_action_merge_0.01/merged_network/OTW/tw_7d-tw_slide_interval_6d/
results/moltbook/top_co_action_merge_None/merged_network/OTW/tw_1d-tw_slide_interval_12h/average/
```

`info_tw/` stores the computed time-window metadata, for example:

```text
info_tw/window_list.p
```

### Single-Layer Co-Action Results

Each co-action layer has its own directory under the time-window root:

```text
<merge_type>/<co-action>/<similarity_function>/
├── edge_list/
│   ├── <merged_start>_<merged_end>.p
│   └── temporal/
│       └── <window_start>_<window_end>.p
├── info_edge_list/
│   ├── <merged_start>_<merged_end>.csv
│   └── temporal/
│       └── <window_start>_<window_end>.csv
├── processed/
└── analysis/
    ├── network_metrics.csv
    ├── nAction_threshold_df.csv
    ├── nAction_layer_df.csv
    ├── nAction_overlapping_df.csv
    ├── plot_nAction_threshold_nNodes.png
    └── plot_nAction_threshold_nEdges.png
```

Example:

```text
results/mh/top_co_action_merge_0.01/merged_network/OTW/tw_7d-tw_slide_interval_6d/average/
└── co-comment/
    └── tfidf_cosine_similarity/
        ├── edge_list/
        ├── info_edge_list/
        ├── processed/
        └── analysis/
```

The main contents are:

```text
edge_list/             Pickled edge lists. The root contains merged edge lists; temporal/ contains one file per window.
info_edge_list/        Optional CSV metadata for edges, mostly useful for overlapping-style and saved-info analyses.
processed/             Intermediate objects created while building or filtering networks.
analysis/              Network-level statistics, threshold-selection CSVs, and plots.
```

### Filtered Network Results

Graph filters are represented as nested path segments below the co-action similarity directory:

```text
<co-action>/<similarity_function>/<filter_1>/<filter_2>/
├── edge_list/
├── edge_list_df/
├── info_edge_list/
├── processed/
├── graph/
├── gephi_graph/
├── analysis/
└── community/
```

Example with a chained filter:

```text
co-comment/tfidf_cosine_similarity/merge_filter_action_2/median_0.914/
```

The first segment, `merge_filter_action_2`, means that edges were kept after applying the action-count filter. The
second segment, `median_0.914`, is a data-driven weight filter. For filters such as `median`, `mean`, `low_std`, and
`high_std`, the threshold value is computed from the previous edge-list output and then used in the directory name.

Filtered directory contents:

```text
edge_list/             Pickled filtered edge lists.
edge_list_df/          CSV version of filtered edge lists.
graph/                 Pickled NetworkX graph objects.
gephi_graph/           GEXF files for Gephi.
analysis/              Metrics and plots computed on the filtered network.
community/             Community-detection roots for this filtered layer.
```

### Multiplex Results

When more than one co-action is selected, the framework creates a multiplex instance under `multi_co_action/`:

```text
<merge_type>/multi_co_action/<compact_layer_instance>/
├── graph/
│   └── multiplex_graph.txt
├── edge_list_df/
├── processed/
├── analysis/
│   ├── network_metrics.csv
│   ├── jaccard_actors_multiplex_graph.csv
│   ├── coverage_edges_multiplex_graph.csv
│   └── *_layer_comparison_heatmaps.png
├── visualization/
├── community/
├── overlapping_analysis/
└── latex/
```

Example:

```text
multi_co_action/url_tics_mfa_2_md_0.908__m_tics_mfa_2_md_0.044__h_tics_mfa_2_md_0.062__rp_tics_mfa_2_md_0.127__rt_tics_mfa_2_md_0.057/
```

The compact multiplex directory name is generated from the selected co-actions, similarity functions, and filters:

```text
url    co-url-domain
m      co-mention
h      co-hashtag
rp     co-reply
rt     co-retweet
tics   tfidf_cosine_similarity
mfa_2  merge_filter_action_2
md_*   median filter with the resolved threshold
```

### Community Detection Results

Community-detection output is stored below the `community/` directory of either a filtered single-layer network or a
multiplex network:

```text
community/<algorithm_repr>/
├── coms/
│   └── coms.p
├── graph/
├── gephi_graph/
├── user_dataframe/
│   └── com_df.csv
├── visualization/
└── analysis/
    ├── <algorithm>_statistics_communities.csv
    ├── <algorithm>_info_cda_per_community.csv
    ├── <algorithm>_info_cda_per_layer.csv
    ├── <algorithm>_coordination_communities.csv
    ├── <algorithm>_group_isControl_validation_communities.csv
    ├── <algorithm>_top_10_communities_summary.csv
    ├── <algorithm>_top_10_communities_url_domain_frequency.csv
    └── <algorithm>_top_10_communities_url_domain_category_percentage.csv
```

`com_df.csv` is the central membership file. For single-layer networks it maps users to communities. For multiplex
networks it maps actor-layer tuples to communities and includes the layer information.

### Community Comparison Results

Community comparisons are written under the multiplex instance `overlapping_analysis/` directory:

```text
overlapping_analysis/
├── <prefix>_overlapping_tensor.p
├── <prefix>_overlapping_set.p
├── <prefix>_single_layer_metrics_communities.csv
├── <prefix>_node_metrics.csv
├── heatmap/
├── stacked_plot/
│   └── flux_df/
├── t_sne_plot/
├── umap_plot/
├── starplot/
├── pca_plot/
├── NMI/
├── node_metrics_gained_lost/
│   ├── KDE_plot/
│   └── distribution_plot/
├── node_metrics_boxplot/
├── validation/
└── cosine_similarity/
```

This directory is used by `CommunityComparisonManager` to compare partitions produced by different algorithms or
different network instances.

### Latex Results

LaTeX tables are written inside the multiplex instance `latex/` directory:

```text
latex/
├── glouvain_top_10_communities_summary_table.tex
└── glouvain_top_10_communities_url_category_table.tex
```

These files are generated from CSVs already stored in the community `analysis/` directory.
