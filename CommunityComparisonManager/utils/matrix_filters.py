from __future__ import annotations

from typing import Any

import numpy as np


class MatrixFilter:
    """
    Filter overlap matrices and format matrix annotations.
    """

    def annotate_format(self, value: Any) -> str:
        """
        Format a heatmap annotation value.

        :param value: [Any] Numeric or string value.
        :return: [str] Formatted annotation.
        """
        if np.issubdtype(type(value), np.integer):
            return f"{int(value)}"
        if np.issubdtype(type(value), np.floating):
            if value.is_integer():
                return f"{int(value)}"
            return f"{value:.3f}"
        return str(value)

    def get_filter_height_matrix_list(self, height_matrix_list: np.ndarray, indices_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter concatenated matrix heights after row filtering.

        :param height_matrix_list: [np.ndarray] Row counts for each vertically concatenated matrix.
        :param indices_y: [np.ndarray] Retained y-axis indices.
        :return: [tuple[np.ndarray, np.ndarray]] Non-empty matrix indices and retained row counts.
        """
        cumulative_rows = np.cumsum(height_matrix_list)
        filter_height_matrix_list = np.zeros_like(height_matrix_list)

        start_idy = 0
        for index, height in enumerate(height_matrix_list):
            end_idy = cumulative_rows[index]
            filter_height_matrix_list[index] = np.sum((indices_y >= start_idy) & (indices_y < end_idy))
            start_idy = end_idy

        no_zero_indices = np.where(filter_height_matrix_list > 0)[0]
        return no_zero_indices, filter_height_matrix_list[no_zero_indices]

    def filter_communities(
        self,
        matrix: np.ndarray,
        community_size_x: np.ndarray,
        community_size_y: np.ndarray,
        communities_label_x: np.ndarray,
        communities_label_y: np.ndarray,
        community_size_th: int | None,
        intersection_matrix: np.ndarray | None = None,
        x_set: np.ndarray | None = None,
        y_set: np.ndarray | None = None,
    ) -> tuple[Any, ...]:
        """
        Filter rows and columns by minimum community size.

        :param matrix: [np.ndarray] Overlap matrix.
        :param community_size_x: [np.ndarray] X community sizes.
        :param community_size_y: [np.ndarray] Y community sizes.
        :param communities_label_x: [np.ndarray] X community labels.
        :param communities_label_y: [np.ndarray] Y community labels.
        :param community_size_th: [int | None] Minimum community size.
        :param intersection_matrix: [np.ndarray | None] Optional intersection-set matrix.
        :param x_set: [np.ndarray | None] Optional x set array.
        :param y_set: [np.ndarray | None] Optional y set array.
        :return: Filtered matrix components.
        """
        if community_size_th is not None:
            indices_x = np.where(community_size_x >= community_size_th)[0]
            indices_y = np.where(community_size_y >= community_size_th)[0]
            filter_community_size_x = community_size_x[indices_x]
            filter_community_size_y = community_size_y[indices_y]
            filter_matrix = matrix[indices_y[:, None], indices_x]
            filter_communities_label_x = communities_label_x[indices_x]
            filter_communities_label_y = communities_label_y[indices_y]

            if intersection_matrix is not None:
                filter_intersection_matrix = intersection_matrix[indices_y[:, None], indices_x]
                filter_x_set = x_set[indices_x]
                filter_y_set = y_set[indices_y]
        else:
            filter_community_size_x = community_size_x
            filter_community_size_y = community_size_y
            filter_matrix = matrix
            filter_communities_label_x = communities_label_x
            filter_communities_label_y = communities_label_y
            filter_intersection_matrix = intersection_matrix
            filter_x_set = x_set
            filter_y_set = y_set

        if intersection_matrix is None:
            return filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y
        return (
            filter_matrix,
            filter_community_size_x,
            filter_community_size_y,
            filter_communities_label_x,
            filter_communities_label_y,
            filter_intersection_matrix,
            filter_x_set,
            filter_y_set,
        )

    def filter_communities_and_height_matrix_list(
        self,
        matrix: np.ndarray,
        community_size_x: np.ndarray,
        community_size_y: np.ndarray,
        communities_label_x: np.ndarray,
        communities_label_y: np.ndarray,
        height_matrix_list: np.ndarray,
        y_label_list: np.ndarray,
        community_size_th: int | None,
    ) -> tuple[Any, ...]:
        """
        Filter overlap matrix data and concatenated heatmap metadata.

        :param matrix: [np.ndarray] Overlap matrix.
        :param community_size_x: [np.ndarray] X community sizes.
        :param community_size_y: [np.ndarray] Y community sizes.
        :param communities_label_x: [np.ndarray] X community labels.
        :param communities_label_y: [np.ndarray] Y community labels.
        :param height_matrix_list: [np.ndarray] Concatenated matrix row counts.
        :param y_label_list: [np.ndarray] Y-axis layer labels.
        :param community_size_th: [int | None] Minimum community size.
        :return: Filtered matrix components and concatenated heatmap metadata.
        """
        if community_size_th is not None:
            indices_y = np.where(community_size_y >= community_size_th)[0]
            no_zero_indices, filter_height_matrix_list = self.get_filter_height_matrix_list(height_matrix_list, indices_y)
            filter_y_label_list = y_label_list[no_zero_indices]
        else:
            filter_height_matrix_list = height_matrix_list
            filter_y_label_list = y_label_list

        filter_matrix, filter_community_size_x, filter_community_size_y, filter_communities_label_x, filter_communities_label_y = self.filter_communities(
            matrix,
            community_size_x,
            community_size_y,
            communities_label_x,
            communities_label_y,
            community_size_th,
        )
        return (
            filter_matrix,
            filter_community_size_x,
            filter_community_size_y,
            filter_communities_label_x,
            filter_communities_label_y,
            filter_height_matrix_list,
            filter_y_label_list,
        )
