from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass
class PipelineConfig:
    """
    External-facing configuration object used by dataset main scripts.
    """
    dataset_name: str
    known_url: list[str] = field(default_factory=list)
    exclude_domain_list: list[str] = field(default_factory=list)
    exclude_hashtag_list: list[str] = field(default_factory=list)
    exclude_mention_dict: dict[str, str] = field(default_factory=dict)
    filter_dataset: dict[str, bool] = field(default_factory=dict)
    user_fraction: float | None = None
    type_filter: str = "top_co_action_merge"
    user_selection_fractions: list[float] = field(default_factory=list)
    tw: Any | None = None
    co_action_list: list[str] = field(default_factory=list)
    similarity_function: str = "tfidf_cosine_similarity"
    list_ca: list[Any] = field(default_factory=list)
    co_action_filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics_to_compute: list[str] = field(default_factory=list)
    metrics_node_to_compute: list[str] = field(default_factory=list)
    single_layer_algorithm_dict: dict[str, list[dict[str, Any] | None]] = field(default_factory=dict)
    multiplex_algorithm_dict: dict[str, list[dict[str, Any] | None]] = field(default_factory=dict)
    text_similarity_threshold: float = 0.7
    text_similarity_chunk_size: int = 5000
    similarity_parallelize_window: int = 1
    elastic_info: dict[str, Any] | None = None

    @property
    def exclude_mention_list(self) -> list[str]:
        """
        Return mention ids to remove from mention-based co-action inputs.

        :return: [list[str]] Mention ids.
        """
        return list(self.exclude_mention_dict.values())


def load_config(dataset_name: str) -> PipelineConfig:
    """
    Load the configuration object for a dataset.

    :param dataset_name: [str] Dataset name, for example moltbook.
    :return: [PipelineConfig] Dataset configuration.
    """
    module = import_module(f"configs.{dataset_name}")
    return module.get_config()
