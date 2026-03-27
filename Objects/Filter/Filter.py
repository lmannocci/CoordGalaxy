from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.common_variables import *
import os

file_name = os.path.splitext(os.path.basename(__file__))[0]


class Filter:
    def __init__(self, type_filter, threshold=None, previous_filter=None):
        self.icm = IntegrityConstraintManager(file_name)
        self.icm.check_filter_graph(type_filter, threshold, previous_filter)
        self.type_filter = type_filter
        self.threshold = threshold
        self.previous_filter = previous_filter

    def get_threshold(self):
        return self.threshold

    def get_previous_filter(self):
        return self.previous_filter

    def get_type_filter(self):
        return self.type_filter

    def set_threshold(self, threshold):
        self.threshold = threshold

    def __str__(self):
        return self.type_filter + "_" + str(self.threshold)

    def __repr__(self):
        previous_filter = self.get_previous_filter()
        filter_concat = ""
        while previous_filter is not None:
            filter_concat = f"{str(previous_filter)}_{filter_concat}"
            previous_filter = previous_filter.get_previous_filter()

        filter_concat = f"{filter_concat}{self.__str__()}"

        return filter_concat
    #
    # def __repr__(self):
    #     tree_filter = ""
    #     previous_filter = self.get_previous_filter()
    #     while previous_filter is not None:
    #         tree_filter = f"{tree_filter}_{str(previous_filter)}"
    #         previous_filter = previous_filter.get_previous_filter()
    #     tree_filter = tree_filter +  self.__str__()
    #     return tree_filter

    def filter_repr_abbr(self):
        filter_ca_str = self.__repr__()
        for key, value in filter_map.items():
            filter_ca_str = filter_ca_str.replace(key, value)
        return filter_ca_str