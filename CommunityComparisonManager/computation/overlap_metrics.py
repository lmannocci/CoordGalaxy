from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


class OverlapMetricCalculator:
    """
    Compute overlap and matching metrics between community member sets.
    """

    def compute_overlap(self, set_x: set[Any], set_y: set[Any], num_decimal: int = 3) -> tuple[dict[str, float | int], set[Any]]:
        """
        Compute overlap metrics between two sets.

        :param set_x: [set[Any]] First community member set.
        :param set_y: [set[Any]] Second community member set.
        :param num_decimal: [int] Number of decimals used for ratio metrics.
        :return: [tuple[dict[str, float | int], set[Any]]] Metric dictionary and intersection set.
        """
        intersection = set_x & set_y
        union = set_x | set_y

        len_x = len(set_x)
        len_y = len(set_y)
        len_intersection = len(intersection)
        len_union = len(union)
        len_min = min(len_x, len_y)

        metrics: dict[str, float | int] = {"absolute": len_intersection}
        metrics["intersect_x"] = round(len_intersection / len_x, num_decimal) if len_x != 0 else 0
        metrics["intersect_y"] = round(len_intersection / len_y, num_decimal) if len_y != 0 else 0
        metrics["minimum"] = round(len_intersection / len_min, num_decimal) if len_min != 0 else 0
        metrics["jaccard"] = round(len_intersection / len_union, num_decimal) if len_union != 0 else 0

        numerator = 2 * metrics["intersect_x"] * metrics["intersect_y"]
        denominator = metrics["intersect_x"] + metrics["intersect_y"]
        metrics["harmonicMean"] = round(numerator / denominator, num_decimal) if denominator != 0 else 0
        return metrics, intersection

    def max_similarity_match(self, similarity_matrix: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int], float]:
        """
        Match rows and columns by maximizing similarity.

        :param similarity_matrix: [np.ndarray] Matrix with row/column similarity scores.
        :return: [tuple[list[tuple[int, int]], list[int], list[int], float]] Matched index pairs, unmatched rows,
            unmatched columns, and total similarity.
        """
        cost_matrix = -similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        total_similarity = similarity_matrix[row_indices, col_indices].sum()
        matches = [(i, j) for i, j in zip(row_indices, col_indices)]
        unmatched_rows = list(set(range(similarity_matrix.shape[0])) - set(row_indices))
        unmatched_cols = list(set(range(similarity_matrix.shape[1])) - set(col_indices))
        return matches, unmatched_rows, unmatched_cols, total_similarity
