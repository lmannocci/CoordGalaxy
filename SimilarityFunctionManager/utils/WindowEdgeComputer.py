from SimilarityFunctionManager.methods.similarityFunction import my_cosine_similarity, overlapping_coefficient
from utils.common_variables import NODE1_VAR, NODE2_VAR, co_action_column, co_action_embeddings

from itertools import combinations
from multiprocessing import Pool
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class WindowEdgeComputer:
    DEFAULT_TEXT_SIMILARITY_MAX_CELLS = 25_000_000

    def __init__(
        self,
        ca: Any,
        sparse_computation: bool,
        save_info: bool,
        parallelize_similarity: bool,
        text_similarity_threshold: float,
        text_similarity_chunk_size: int | None = None
    ) -> None:
        """
            Create a per-window edge computer for one co-action and similarity configuration.
            :param ca: [CoAction] Co-action and similarity-function configuration.
            :param sparse_computation: [bool] Whether to use sparse matrix computation where supported.
            :param save_info: [bool] Whether to return object-level edge-contribution details.
            :param parallelize_similarity: [bool] Whether to parallelize pairwise user similarity inside a window.
            :param text_similarity_threshold: [float] Minimum similarity for text-pair matches.
            :param text_similarity_chunk_size: [int | None] Number of text rows per matrix-multiplication chunk.
                If None, the chunk size is chosen from the window size to keep the similarity matrix bounded.
            :return: None.
        """
        self.ca = ca
        self.sparse_computation = sparse_computation
        self.save_info = save_info
        self.parallelize_similarity = parallelize_similarity
        self.text_similarity_threshold = text_similarity_threshold
        self.text_similarity_chunk_size = text_similarity_chunk_size
        self.text_embeddings: np.ndarray | None = None

    def set_text_embeddings(self, text_embeddings: np.ndarray) -> None:
        """
            Store text embeddings aligned with the row_idx column of text co-action dataframes.
            :param text_embeddings: [np.ndarray] Embedding matrix aligned to the co-action CSV row order.
            :return: None.
        """
        self.text_embeddings = text_embeddings

    def compute(self, df: pd.DataFrame) -> tuple[list[tuple], pd.DataFrame]:
        """
            Compute the edge list for one time-window dataframe.
            :param df: [pd.DataFrame] Window dataframe for the configured co-action.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and optional edge-contribution dataframe.
        """
        if self.ca.get_co_action() in co_action_embeddings:
            return self._compute_text_window_edges(df)
        return self._compute_object_window_edges(df)

    def _compute_text_window_edges(self, df: pd.DataFrame) -> tuple[list[tuple], pd.DataFrame]:
        """
            Compute edges for a text-embedding co-action inside one time window.
            :param df: [pd.DataFrame] Window dataframe with userId, id, and row_idx columns.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and optional matched-text dataframe.
        """
        if self.text_embeddings is None:
            raise ValueError("Text embeddings must be set before computing text co-action edges.")

        if df.empty:
            return [], pd.DataFrame()

        # The old text implementation iterated over every pair of users and then over every text of user A
        # against every text of user B. That preserves exact semantics, but the heavy work happens in Python
        # loops. Here we invert the computation: inside one time window we compare text embeddings directly,
        # keep only text-text pairs above the threshold, and then aggregate those matches back to user-user
        # edges. The result is still exact for the configured threshold; only the execution strategy changes.
        text_df = df[["userId", "id", "row_idx"]].copy().reset_index(drop=True)
        row_indices = text_df["row_idx"].to_numpy(dtype=np.int64)
        user_ids = text_df["userId"].astype(str).to_numpy()
        content_ids = text_df["id"].astype(str).to_numpy()

        # Embeddings are created with normalize_embeddings=True in InputManager/Content/embedding.py.
        # For normalized vectors, cosine similarity equals the dot product. The matrix multiplication below
        # therefore computes all cosine similarities between one chunk and the full window at once.
        window_embeddings = np.asarray(self.text_embeddings[row_indices], dtype=np.float32)
        chunk_size = self._text_similarity_chunk_size(n_texts=window_embeddings.shape[0])
        edge_accumulator: dict[tuple[str, str], dict[str, Any]] = {}
        info_records: list[tuple[str, str, str, str]] = []

        for chunk_start in range(0, window_embeddings.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, window_embeddings.shape[0])
            chunk_embeddings = window_embeddings[chunk_start:chunk_end]

            # sim_matrix has shape (chunk_size, n_texts_in_window). Entry (i, j) is the cosine
            # similarity between chunk text i and window text j. This matrix is the expensive object, so
            # we build only one chunk at a time instead of a full n_texts x n_texts matrix.
            sim_matrix = chunk_embeddings @ window_embeddings.T
            local_rows, matched_cols, matched_sims = self._threshold_text_similarity_chunk(
                sim_matrix=sim_matrix,
                chunk_start=chunk_start,
                user_ids=user_ids
            )

            self._accumulate_text_similarity_matches(
                edge_accumulator=edge_accumulator,
                info_records=info_records,
                chunk_start=chunk_start,
                local_rows=local_rows,
                matched_cols=matched_cols,
                matched_sims=matched_sims,
                user_ids=user_ids,
                content_ids=content_ids
            )

        return self._build_text_edge_output(edge_accumulator, info_records)

    def _text_similarity_chunk_size(self, n_texts: int) -> int:
        """
            Return the number of text embeddings to process in one matrix-multiplication chunk.
            :param n_texts: [int] Number of text rows in the current time window.
            :return: [int] Chunk size used for chunked text similarity computation.
        """
        if self.text_similarity_chunk_size is not None:
            return max(1, min(self.text_similarity_chunk_size, n_texts))

        # The chunk produces a matrix with chunk_size * n_texts float32 cells. The default cap keeps
        # that matrix around 100 MB: 25,000,000 cells * 4 bytes. This is only a heuristic; callers can
        # pass text_similarity_chunk_size when they know their memory budget better.
        return max(1, min(n_texts, self.DEFAULT_TEXT_SIMILARITY_MAX_CELLS // max(1, n_texts)))

    def _threshold_text_similarity_chunk(
        self,
        sim_matrix: np.ndarray,
        chunk_start: int,
        user_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Select cross-user text pairs above the text similarity threshold from one similarity chunk.
            :param sim_matrix: [np.ndarray] Similarity matrix shaped as chunk rows by all window rows.
            :param chunk_start: [int] Global window-row index of the first chunk row.
            :param user_ids: [np.ndarray] User id for each text row in the current time window.
            :return: [tuple[np.ndarray, np.ndarray, np.ndarray]] Local chunk rows, matched columns, and similarities.
        """
        n_rows, n_cols = sim_matrix.shape
        local_row_indices = np.arange(n_rows)[:, None]
        global_row_indices = local_row_indices + chunk_start
        column_indices = np.arange(n_cols)[None, :]

        # Keep only the upper triangle of the full window similarity matrix by requiring col > row.
        # This removes self-comparisons and prevents the symmetric pair (j, i) from being counted again
        # after (i, j) has already been considered.
        upper_triangle_mask = column_indices > global_row_indices

        # Text pairs from the same user do not create user-user co-action edges, so they must be removed
        # before aggregation. Broadcasting compares each chunk-row user against every text user in window.
        cross_user_mask = user_ids[chunk_start:chunk_start + n_rows, None] != user_ids[None, :]

        # The threshold is finally applied at the matrix level. This is where the number of retained pairs
        # becomes small; unlike the old implementation, the thresholded output is what we iterate in Python.
        threshold_mask = sim_matrix >= self.text_similarity_threshold
        matched_rows, matched_cols = np.nonzero(upper_triangle_mask & cross_user_mask & threshold_mask)
        matched_sims = sim_matrix[matched_rows, matched_cols].astype(float, copy=False)
        return matched_rows, matched_cols, matched_sims

    def _accumulate_text_similarity_matches(
        self,
        edge_accumulator: dict[tuple[str, str], dict[str, Any]],
        info_records: list[tuple[str, str, str, str]],
        chunk_start: int,
        local_rows: np.ndarray,
        matched_cols: np.ndarray,
        matched_sims: np.ndarray,
        user_ids: np.ndarray,
        content_ids: np.ndarray
    ) -> None:
        """
            Aggregate matched text pairs into user-user edge statistics.
            :param edge_accumulator: [dict[tuple[str, str], dict[str, Any]]] Running user-pair statistics.
            :param info_records: [list[tuple[str, str, str, str]]] Running matched text-pair records for save_info.
            :param chunk_start: [int] Global window-row index of the first chunk row.
            :param local_rows: [np.ndarray] Local chunk-row indices of matched text pairs.
            :param matched_cols: [np.ndarray] Global window-column indices of matched text pairs.
            :param matched_sims: [np.ndarray] Similarity values for the matched text pairs.
            :param user_ids: [np.ndarray] User id for each text row in the current time window.
            :param content_ids: [np.ndarray] Content id for each text row in the current time window.
            :return: None. edge_accumulator and info_records are updated in place.
        """
        for local_row, matched_col, sim in zip(local_rows, matched_cols, matched_sims):
            global_row = chunk_start + int(local_row)
            global_col = int(matched_col)
            user_id_1 = user_ids[global_row]
            user_id_2 = user_ids[global_col]

            # We keep a stable undirected edge key so all text matches between the same two users aggregate
            # into one edge, regardless of which user's text appears first in the dataframe window.
            edge_key = (user_id_1, user_id_2) if user_id_1 <= user_id_2 else (user_id_2, user_id_1)
            edge_stats = edge_accumulator.setdefault(edge_key, {"sum": 0.0, "count": 0})
            edge_stats["sum"] += float(sim)
            edge_stats["count"] += 1

            if self.save_info:
                if edge_key == (user_id_1, user_id_2):
                    info_records.append((user_id_1, user_id_2, content_ids[global_row], content_ids[global_col]))
                else:
                    info_records.append((user_id_2, user_id_1, content_ids[global_col], content_ids[global_row]))

    def _build_text_edge_output(
        self,
        edge_accumulator: dict[tuple[str, str], dict[str, Any]],
        info_records: list[tuple[str, str, str, str]]
    ) -> tuple[list[tuple], pd.DataFrame]:
        """
            Convert aggregated text-match statistics to the framework edge-list output format.
            :param edge_accumulator: [dict[tuple[str, str], dict[str, Any]]] Aggregated user-pair statistics.
            :param info_records: [list[tuple[str, str, str, str]]] Matched text pairs retained when save_info is True.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and optional matched-text dataframe.
        """
        edge_list = []
        for (user_id_1, user_id_2), edge_stats in edge_accumulator.items():
            n_similar_texts = edge_stats["count"]
            avg_sim = edge_stats["sum"] / n_similar_texts
            edge_list.append((user_id_1, user_id_2, avg_sim, n_similar_texts))

        if self.save_info:
            info_edge_list_df = pd.DataFrame(
                info_records,
                columns=[NODE1_VAR, NODE2_VAR, "id_1", "id_2"]
            )
        else:
            info_edge_list_df = pd.DataFrame()

        return edge_list, info_edge_list_df

    def _compute_object_window_edges(self, df: pd.DataFrame) -> tuple[list[tuple], pd.DataFrame]:
        """
            Compute edges for a set/object co-action inside one time window.
            :param df: [pd.DataFrame] Window dataframe with userId and objectId columns.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and optional object-contribution dataframe.
        """
        c = co_action_column[self.ca.get_co_action()]
        co_action_df = df.groupby("userId")[c].apply(set).reset_index()

        if self.sparse_computation:
            return self._compute_sparse_object_edges(co_action_df, c)
        return self._compute_dense_object_edges(co_action_df, c)

    def _compute_sparse_object_edges(self, co_action_df: pd.DataFrame, c: str) -> tuple[list[tuple], pd.DataFrame]:
        """
            Compute object co-action edges using sparse TF-IDF cosine similarity.
            :param co_action_df: [pd.DataFrame] One row per user with a set of object ids.
            :param c: [str] Name of the object-set column.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and an empty info dataframe.
        """
        edge_list = []
        if self.ca.get_similarity_function() != "tfidf_cosine_similarity":
            return edge_list, pd.DataFrame()

        user_ids = co_action_df["userId"].values.tolist()
        tfidf_matrix = self._tf_idf(co_action_df, c)
        sim_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
        upper_tri_indices = np.triu_indices(sim_matrix.shape[0], k=1)

        for i, j in zip(*upper_tri_indices):
            sim = sim_matrix[i, j]
            if sim > 0:
                edge_list.append((user_ids[i], user_ids[j], sim, None))

        return edge_list, pd.DataFrame()

    def _compute_dense_object_edges(self, co_action_df: pd.DataFrame, c: str) -> tuple[list[tuple], pd.DataFrame]:
        """
            Compute object co-action edges by comparing every pair of users.
            :param co_action_df: [pd.DataFrame] One row per user with a set of object ids.
            :param c: [str] Name of the object-set column.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and optional object-contribution dataframe.
        """
        co_action_sets = self._build_object_user_records(co_action_df, c)
        user_pairs = combinations(co_action_sets, 2)
        if not self.parallelize_similarity:
            results = [self._compute_object_pair(pair) for pair in user_pairs]
        else:
            with Pool() as pool:
                results = pool.map(self._compute_object_pair, user_pairs)
        return self._collect_object_results(results)

    def _build_object_user_records(self, co_action_df: pd.DataFrame, c: str) -> list[tuple]:
        """
            Build one object-set tuple per user for dense pairwise computation.
            :param co_action_df: [pd.DataFrame] One row per user with a set of object ids.
            :param c: [str] Name of the object-set column.
            :return: [list[tuple]] User tuples for pairwise similarity computation.
        """
        user_ids = co_action_df["userId"].values.tolist()
        if self.ca.get_similarity_function() == "tfidf_cosine_similarity":
            tfidf_matrix = self._tf_idf(co_action_df, c)
            mat = tfidf_matrix.toarray()
            list_user_sets = list(co_action_df[c].values)
            return list(zip(user_ids, list_user_sets, mat))

        if self.ca.get_similarity_function() in ["overlapping_coefficient", "overlapping"]:
            return list(co_action_df.to_records(index=False))

        return []

    def _compute_object_pair(self, pair: tuple[Any, Any]) -> tuple | None:
        """
            Compute one pairwise object-set edge.
            :param pair: [tuple[Any, Any]] Two user tuples containing userId, object set, and optionally TF-IDF vector.
            :return: [tuple | None] Edge tuple with intersection details, or None when similarity is zero.
        """
        user_tuple1, user_tuple2 = pair
        userId1 = user_tuple1[0]
        userId2 = user_tuple2[0]
        set1 = user_tuple1[1]
        set2 = user_tuple2[1]
        sim = 0
        nCommonAction = 0
        intersection = set()

        if self.ca.get_similarity_function() == "tfidf_cosine_similarity":
            v1 = user_tuple1[2].reshape(1, -1)
            v2 = user_tuple2[2].reshape(1, -1)
            sim = my_cosine_similarity(v1, v2)
            intersection, nCommonAction, _ = overlapping_coefficient(set1, set2)
        elif self.ca.get_similarity_function() == "overlapping_coefficient":
            intersection, nCommonAction, sim = overlapping_coefficient(set1, set2)
        elif self.ca.get_similarity_function() == "overlapping":
            intersection, nCommonAction, _ = overlapping_coefficient(set1, set2)
            sim = nCommonAction

        if sim <= 0:
            return None
        return userId1, userId2, sim, nCommonAction, list(intersection)

    def _collect_object_results(self, results: Iterable[tuple | None]) -> tuple[list[tuple], pd.DataFrame]:
        """
            Collect pairwise results for object co-actions.
            :param results: [Iterable[tuple | None]] Results returned by _compute_object_pair.
            :return: [tuple[list[tuple], pd.DataFrame]] Edge list and optional object-contribution dataframe.
        """
        edge_list = []
        intersection_list = []
        userId_list1 = []
        userId_list2 = []

        for result in results:
            if result is None:
                continue
            userId1, userId2, sim, nCommonAction, intersection = result
            edge_list.append((userId1, userId2, sim, nCommonAction))

            if self.save_info:
                intersection_list.extend(intersection)
                userId_list1.extend([userId1] * len(intersection))
                userId_list2.extend([userId2] * len(intersection))

        if self.save_info:
            info_edge_list_df = pd.DataFrame(
                {NODE1_VAR: userId_list1, NODE2_VAR: userId_list2, 'id': intersection_list})
        else:
            info_edge_list_df = pd.DataFrame()

        return edge_list, info_edge_list_df

    def _tf_idf(self, co_action_df: pd.DataFrame, c: str) -> Any:
        """
            Build a TF-IDF matrix from each user's set of co-action objects.
            :param co_action_df: [pd.DataFrame] Dataframe with one row per user and a set/list column of objects.
            :param c: [str] Name of the object-set column.
            :return: Sparse TF-IDF matrix with one row per user.
        """
        documents = co_action_df[c].apply(self._join_objects).values.tolist()
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(documents)

    def _join_objects(self, values: Iterable[Any]) -> str:
        """
            Join one user's object ids into one TF-IDF document.
            :param values: [Iterable[Any]] Object ids for one user.
            :return: [str] Space-separated document.
        """
        return " ".join(str(value) for value in values)
