import os
from typing import Any

from IntegrityConstraintManager.IntegrityConstraintManager import IntegrityConstraintManager
from utils.common_variables import filter_map


file_name = os.path.splitext(os.path.basename(__file__))[0]


class Filter:
    def __init__(
        self,
        type_filter: str,
        threshold: int | float | None = None,
        previous_filter: Any | None = None,
    ) -> None:
        """
            Create a graph-filter configuration.
            :param type_filter: [str] Filter strategy name, for example merge_filter_action, th, median, or backbone.
            :param threshold: [int | float | None] Threshold used by the filter. Use None for data-driven filters
                median, mean, low_std, and high_std because they compute the value from the input edge list.
            :param previous_filter: [Filter | None] Previous filter in the chain. Use this when the current filter
                must read the output of an earlier filter.
            :return: None.
        """
        self.icm = IntegrityConstraintManager(file_name)
        self.icm.check_filter_graph(type_filter, threshold, previous_filter)
        self.type_filter = type_filter
        self.threshold = threshold
        self.previous_filter = previous_filter

    def get_threshold(self) -> int | float | None:
        """
            Return the configured threshold.
            :return: [int | float | None] Threshold value used by the filter.
        """
        return self.threshold

    def get_previous_filter(self) -> Any | None:
        """
            Return the previous filter in the chain.
            :return: [Filter | None] Previous filter, or None when this is the first filter.
        """
        return self.previous_filter

    def get_type_filter(self) -> str:
        """
            Return the filter strategy name.
            :return: [str] Filter strategy name.
        """
        return self.type_filter

    def set_threshold(self, threshold: int | float) -> None:
        """
            Update the threshold after it has been computed from data.
            :param threshold: [int | float] New threshold value.
            :return: None.
        """
        self.threshold = threshold

    def __str__(self) -> str:
        """
            Return the single-filter path representation.
            :return: [str] Filter string in the format type_threshold.
        """
        return self.type_filter + "_" + str(self.threshold)

    def __repr__(self) -> str:
        """
            Return the full chained-filter representation.
            :return: [str] Filter string including previous filters in execution order.
        """
        previous_filter = self.get_previous_filter()
        filter_concat = ""
        while previous_filter is not None:
            filter_concat = f"{str(previous_filter)}_{filter_concat}"
            previous_filter = previous_filter.get_previous_filter()
        return f"{filter_concat}{self.__str__()}"

    def filter_repr_abbr(self) -> str:
        """
            Return an abbreviated representation for compact output paths.
            :return: [str] Chained-filter representation with strategy names replaced by abbreviations.
        """
        filter_ca_str = self.__repr__()
        for key, value in filter_map.items():
            filter_ca_str = filter_ca_str.replace(key, value)
        return filter_ca_str
