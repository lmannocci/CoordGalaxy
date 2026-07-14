from SimilarityFunctionManager.utils import SimilarityComputationLogger, WindowEdgeComputer
from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.Checkpoint.Checkpoint import *
from utils.EdgeListManager.EdgeListManager import EdgeListManager
from utils.common_variables import co_action_column, co_action_embeddings, co_action_map, dtype
from MergeNetworkManager import MergeNetworkManager
from Objects.TimeWindow.TimeWindow import *
from utils.decorator_definition import *

from multiprocessing import Pool, Manager
from functools import partial
import os
# import shutil
import multiprocessing
import time
import numpy as np
import pandas as pd
from typing import Any, Optional

absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class SimilarityFunctionManager:

    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str,
        tw: TimeWindow,
        ca: Any,
        text_similarity_threshold: Optional[float] = None,
        sparse_computation: bool = False,
        save_info: bool = False,
        parallelize_window: bool | int = False,
        parallelize_similarity: bool = False,
        merge_info_edge_list: bool = False,
        text_similarity_chunk_size: int | None = None
    ) -> None:
        """
            Create the similarity manager for one co-action and time-window configuration.
            :param dataset_name: [str] Dataset directory name.
            :param user_fraction: [float | None] User-selection fraction. None reads the base co_action_data directory.
            :param type_filter: [str] User-selection strategy name used to resolve selected co-action data.
            :param tw: [TimeWindow] Time-window configuration.
            :param ca: [CoAction] Co-action and similarity-function configuration.
            :param text_similarity_threshold: [float | None] Minimum embedding similarity for text co-actions.
            :param sparse_computation: [bool] If True, use sparse matrix computation where supported.
            :param save_info: [bool] If True, save object-level edge-contribution details.
            :param parallelize_window: [bool | int] False for serial windows, True for all CPUs, or an integer process count.
            :param parallelize_similarity: [bool] If True, parallelize pairwise user similarity inside each window.
            :param merge_info_edge_list: [bool] If True and save_info is True, merge temporal info-edge CSV files for
                merged output networks. This can produce very large files.
            :param text_similarity_chunk_size: [int | None] Number of text rows per chunk for embedding-based co-actions.
                None lets WindowEdgeComputer choose a memory-bounded chunk size.
            :return: None.
        """
        self.lm = LogManager('main')
        self.similarity_logger = SimilarityComputationLogger(self.lm)
        self.icm = IntegrityConstraintManager(file_name)
        self.dm = DirectoryManager(file_name, dataset_name, data_path=data_path, results=results,
                                   user_fraction=user_fraction, type_filter=type_filter, tw=tw, ca=ca)

        self.ch = Checkpoint()
        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.ca = ca
        self.text_similarity_threshold = text_similarity_threshold if text_similarity_threshold is not None else 0.0
        self.sparse_computation = sparse_computation
        self.save_info = save_info  # if sparse_computation = True, save_info not implemented
        self.merge_info_edge_list = merge_info_edge_list
        self.text_similarity_chunk_size = text_similarity_chunk_size
        self._set_parallelize_window(parallelize_window)
        self.parallelize_similarity = parallelize_similarity

        # check if the chosen sparse_computation is implemented for the chosen similarity function
        self.icm.check_sparse_computation(ca, sparse_computation, save_info, parallelize_similarity)

        self.edge_list_manager = EdgeListManager()
        self.window_edge_computer = WindowEdgeComputer(
            ca=ca,
            sparse_computation=sparse_computation,
            save_info=save_info,
            parallelize_similarity=parallelize_similarity,
            text_similarity_threshold=self.text_similarity_threshold,
            text_similarity_chunk_size=text_similarity_chunk_size
        )

        self.mm = MergeNetworkManager(self.dm, dataset_name, user_fraction, type_filter, tw, ca)

    def _set_parallelize_window(self, parallelize_window: bool | int) -> None:
        """
            Configure time-window parallelization.
            :param parallelize_window: [bool | int] False for serial execution, True for CPU count, or integer process count.
            :return: None.
        """
        if isinstance(parallelize_window, bool):
            self.parallelize_window = parallelize_window
            if parallelize_window:
                # Determine the number of processes to use
                self.num_processes = multiprocessing.cpu_count()
            else:
                self.num_processes = 1
        elif isinstance(parallelize_window, int):
            self.num_processes = parallelize_window
            self.parallelize_window = True
        else:
            self.num_processes = 1
            self.parallelize_window = False
        mode = "parallel" if self.parallelize_window else "serial"
        self.lm.printl(
            f"[SIM][CONFIG] window_execution={mode} requested_parallelize_window={parallelize_window} "
            f"processes={self.num_processes}"
        )

    def _computing_time_window_similarity(self, df: pd.DataFrame) -> None:
        """
            Compute the edge list for every time window in serial or parallel window execution.
            :param df: [pd.DataFrame] Co-action dataframe used to compute temporal edge lists.
            :return: None. Edge lists are saved in the DirectoryManager edge-list paths.
        """
        window_list = self.tw.compute_time_windows(df, self.dm.path_info_tw)
        self.n_windows = len(window_list)
        self.similarity_logger.log_window_plan(
            window_list=window_list,
            parallelize_window=self.parallelize_window,
            num_processes=self.num_processes,
            edge_output_path=self.dm.path_edge_list_temporal
        )
        # delete the dataframe to free memory, once I split it according to the time window
        del df

        if not self.parallelize_window:
            self._compute_windows_serial(window_list)
        else:
            self._compute_windows_parallel(window_list)

        # if the type of output is merged (w.r.t. the temporal axis) I have to merge the edges among the time windows, outputting one edge_list
        if self.tw.get_type_output_network() == 'merged':
            merge_info_edge_list = self._should_merge_info_edge_list()
            self.similarity_logger.log_merge_start(
                temporal_edge_path=self.dm.path_edge_list_temporal,
                merged_edge_path=self.dm.path_edge_list,
                merge_type=self.tw.get_type_merge(),
                merge_info_edge_list=merge_info_edge_list
            )
            self.mm.merge_edge_list(
                self.dm.path_edge_list_temporal,
                self.dm.path_edge_list,
                merge_info_edge_list=merge_info_edge_list
            )
            self.similarity_logger.log_merge_done(self.dm.path_edge_list)

    def _should_merge_info_edge_list(self) -> bool:
        """
            Return whether temporal info-edge CSV files should be merged.
            :return: [bool] True only when save_info and merge_info_edge_list are both enabled.
        """
        return self.save_info and self.merge_info_edge_list

    def _compute_windows_serial(self, window_list: list[Any]) -> None:
        """
            Compute all time windows in the current process.
            :param window_list: [list[Any]] Time-window records returned by TimeWindow.compute_time_windows.
            :return: None. Edge lists are saved to disk.
        """
        for window_index, window in enumerate(window_list, start=1):
            self._window_edge_list(window, window_index)

    def _compute_windows_parallel(self, window_list: list[Any]) -> None:
        """
            Compute time windows in parallel processes.
            :param window_list: [list[Any]] Time-window records returned by TimeWindow.compute_time_windows.
            :return: None. Edge lists are saved to disk.
        """
        with Manager() as manager:
            started_counter = manager.Value('i', 0)
            completed_counter = manager.Value('i', 0)
            counter_lock = manager.Lock()
            with Pool(processes=self.num_processes) as pool:
                pool.map(
                    partial(
                        self._window_edge_list,
                        window_counter=started_counter,
                        completed_counter=completed_counter,
                        counter_lock=counter_lock
                    ),
                    window_list
                )

    def _window_edge_list(
        self,
        window: tuple,
        window_counter: int | Any,
        completed_counter: Any | None = None,
        counter_lock: Any | None = None
    ) -> tuple[str, str]:
        """
            Compute the edge list for the given time window.
            :param window: [tuple] Time-window tuple containing start/end metadata and the filtered dataframe.
            :param window_counter: [int | Any] Serial window index or shared integer-like process value.
            :param completed_counter: [Any | None] Shared completed-window counter for parallel execution.
            :param counter_lock: [Any | None] Shared lock used to update parallel counters consistently.
            :return: [tuple[str, str]] Start and end date labels for the processed window.
        """
        start_date = window[2]
        end_date = window[3]
        df = window[4]
        window_index = self._increment_window_counter(window_counter, counter_lock)
        window_started_at = time.time()
        process_id = os.getpid()
        self.similarity_logger.log_window_start(
            window_index=window_index,
            n_windows=self.n_windows,
            start_date=start_date,
            end_date=end_date,
            df=df,
            process_id=process_id,
            co_action=self.ca.get_co_action()
        )

        edge_list = []
        output_path = None
        info_output_path = None
        if df.shape[0] > 0:
            # name of the file of the edge list
            # I replace : characters because it is bad read both on windows and mac
            filename = f"{start_date.replace(':', '-')}_{end_date.replace(':', '-')}.p"

            edge_list, info_edge_list_df = self.window_edge_computer.compute(df)
            # save edge_list
            output_path = self.dm.path_edge_list_temporal + filename
            self.ch.save_object(edge_list, output_path)

            # info_edge_list are very large files. it is better not to save if it is not necessary
            if self.save_info:
                info_filename = filename.split('.')[0] + '.csv'
                info_output_path = self.dm.path_info_edge_list_temporal + info_filename
                info_edge_list_df.to_csv(info_output_path, index=False)

        completed_index = self._increment_completed_counter(completed_counter, counter_lock, window_index)
        elapsed_seconds = time.time() - window_started_at
        self.similarity_logger.log_window_done(
            window_index=window_index,
            completed_index=completed_index,
            n_windows=self.n_windows,
            start_date=start_date,
            end_date=end_date,
            n_edges=len(edge_list),
            elapsed_seconds=elapsed_seconds,
            process_id=process_id,
            output_path=output_path,
            info_output_path=info_output_path
        )

        return start_date, end_date

    def _increment_window_counter(self, window_counter: int | Any, counter_lock: Any | None = None) -> int:
        """
            Return the current serial window index or increment a shared multiprocessing counter.
            :param window_counter: [int | Any] Serial index or shared integer-like process value.
            :param counter_lock: [Any | None] Shared lock used for parallel counter increments.
            :return: [int] Current window index.
        """
        if isinstance(window_counter, int):
            return window_counter
        if counter_lock is None:
            window_counter.value += 1
            return window_counter.value
        with counter_lock:
            window_counter.value += 1
            return window_counter.value

    def _increment_completed_counter(
        self,
        completed_counter: Any | None,
        counter_lock: Any | None,
        fallback_index: int
    ) -> int:
        """
            Return the completed-window count for serial or parallel execution.
            :param completed_counter: [Any | None] Shared completed-window counter for parallel execution.
            :param counter_lock: [Any | None] Shared lock used for parallel counter increments.
            :param fallback_index: [int] Serial window index used when no completed counter exists.
            :return: [int] Completed-window count.
        """
        if completed_counter is None:
            return fallback_index
        if counter_lock is None:
            completed_counter.value += 1
            return completed_counter.value
        with counter_lock:
            completed_counter.value += 1
            return completed_counter.value

    def _read_co_action_dataset(self) -> pd.DataFrame:
        """
            Get the filtered dataframe, removing missing values for the correct columns of the dataset.
            :return: [pd.DataFrame] Co-action dataframe ready for time-window computation.
        """
        ca_type = self.ca.get_co_action()
        co_action_path = self._co_action_file_path(co_action_map[ca_type], "csv")
        df = self.ch.read_dataframe(co_action_path, dtype)
        rows_before_filter = df.shape[0]
        if co_action_column[ca_type] not in df.columns:
            m = f"{co_action_column[ca_type]} column is not among the columns of the dataframe."
            self.lm.printl(m)
            raise ValueError(m)

        # Remove null values
        df = df[df[co_action_column[ca_type]].isnull() == False]

        df = df.reset_index(drop=True)

        if ca_type in co_action_embeddings:
            embedding_path = self._co_action_file_path(co_action_map[ca_type], "npy")
            embedding = self.ch.load_object(embedding_path)

            if len(df) != len(embedding):
                m = (
                    f"Mismatch between filtered dataframe rows ({len(df)}) "
                    f"and embedding rows ({len(embedding)})."
                )
                self.lm.printl(m)
                raise ValueError(m)

            # Keep only a row index in the dataframe
            df["row_idx"] = np.arange(len(df), dtype=np.int64)

            self.window_edge_computer.set_text_embeddings(np.asarray(embedding, dtype=np.float32))

        self.similarity_logger.log_data_loaded(
            co_action_path=co_action_path,
            rows_before_filter=rows_before_filter,
            rows_after_filter=df.shape[0],
            n_users=df["userId"].nunique()
        )
        return df

    def _co_action_data_path(self) -> str:
        """
            Return the directory to read co-action artifacts from.
            :return: [str] Selected-user co-action directory when present, otherwise the base co_action_data directory.
        """
        filtered_path = self.dm.get_co_action_data_path(self.user_fraction, self.type_filter)
        if self.user_fraction is not None and os.path.isdir(filtered_path):
            return filtered_path
        return self.dm.path_co_action_data

    def _co_action_file_path(self, action_name: str, extension: str) -> str:
        """
            Return the co-action file path, preferring the dataset-directory scoped filename template.
            :param action_name: [str] Co-action filename stem.
            :param extension: [str] File extension without dot.
            :return: [str] Path to the co-action artifact.
        """
        co_action_data_path = self._co_action_data_path()
        filename = f"{action_name}.{extension}"
        path = f"{co_action_data_path}{filename}"
        if os.path.exists(path):
            return path

        legacy_filename = f"{self.dataset_name}_{action_name}.{extension}"
        legacy_path = f"{co_action_data_path}{legacy_filename}"
        if os.path.exists(legacy_path):
            return legacy_path

        if self.user_fraction is not None:
            old_filtered_filename = f"{self.user_fraction}_{self.type_filter}_{action_name}.{extension}"
            old_filtered_path = f"{self.dm.path_co_action_data}{old_filtered_filename}"
            if os.path.exists(old_filtered_path):
                return old_filtered_path

            old_legacy_filename = f"{self.user_fraction}_{self.type_filter}_{self.dataset_name}_{action_name}.{extension}"
            old_legacy_path = f"{self.dm.path_co_action_data}{old_legacy_filename}"
            if os.path.exists(old_legacy_path):
                return old_legacy_path

        return legacy_path

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def compute_similarity(self) -> None:
        """
            Compute the edge list for all time windows in serial or parallel window execution.
            :return: None. Edge-list artifacts are saved to the configured output directories.
        """
        start_time = time.time()

        self.similarity_logger.log_run_start(self)
        df = self._read_co_action_dataset()

        self._computing_time_window_similarity(df)

        finish_time = time.time()
        delta_time = finish_time - start_time
        self.similarity_logger.log_run_done(self.dataset_name, self.ca, delta_time)
    
    # If you converted the ids in the normalization step, it is not necessary to convert them again in the edge-list files. 
    # The following method is commented out, but it can be used if needed, if you forgot to convert the ids in the normalization step.
    @log_method
    def convert_ids_edge_list(self) -> None:
        """
            Convert userId1 and userId2 in saved edge-list files to simple framework ids.
            :return: None. Edge-list pickle files are overwritten in place.
        """
        self.edge_list_manager.convert_ids_directory(self.dm.path_edge_list_temporal)
        self.edge_list_manager.convert_ids_directory(self.dm.path_edge_list)
