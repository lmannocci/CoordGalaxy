from typing import Any

import pandas as pd

from utils.common_variables import action_map


class SimilarityComputationLogger:
    def __init__(self, lm: Any) -> None:
        """
            Create a logger helper for similarity-computation progress messages.
            :param lm: [LogManager] Log manager used by the main similarity manager.
            :return: None.
        """
        self.lm = lm

    def log_run_start(self, manager: Any) -> None:
        """
            Print the run-level similarity configuration.
            :param manager: [SimilarityFunctionManager] Similarity manager whose current configuration is logged.
            :return: None.
        """
        window_mode = "parallel" if manager.parallelize_window else "serial"
        similarity_mode = "parallel" if manager.parallelize_similarity else "serial"
        self.lm.printl(
            "\n".join([
                "[SIM][RUN START]",
                f"  dataset={manager.dataset_name}",
                f"  co_action={manager.ca.get_co_action()} layer={action_map[manager.ca.get_co_action()]}",
                f"  similarity_function={manager.ca.get_similarity_function()}",
                f"  user_fraction={manager.user_fraction} type_filter={manager.type_filter}",
                f"  time_window_type={manager.tw.get_type_time_window()} output={manager.tw.get_type_output_network()}",
                f"  tw={manager.tw.get_tw()} tw_slide={manager.tw.get_tw_slide_interval()} merge={manager.tw.get_type_merge()}",
                f"  window_execution={window_mode} processes={manager.num_processes}",
                f"  pairwise_similarity_execution={similarity_mode}",
                f"  sparse_computation={manager.sparse_computation} save_info={manager.save_info}",
                f"  merge_info_edge_list={manager.merge_info_edge_list}",
                f"  text_similarity_threshold={manager.text_similarity_threshold}",
                f"  text_similarity_chunk_size={manager.text_similarity_chunk_size}",
            ])
        )

    def log_data_loaded(
        self,
        co_action_path: str,
        rows_before_filter: int,
        rows_after_filter: int,
        n_users: int
    ) -> None:
        """
            Print the loaded co-action dataset summary.
            :param co_action_path: [str] Path of the loaded co-action CSV.
            :param rows_before_filter: [int] Number of rows before removing null objects.
            :param rows_after_filter: [int] Number of rows after removing null objects.
            :param n_users: [int] Number of distinct users after filtering.
            :return: None.
        """
        self.lm.printl(
            "[SIM][DATA] "
            f"path={co_action_path} rows_before_null_filter={rows_before_filter} "
            f"rows_after_null_filter={rows_after_filter} users={n_users}"
        )

    def log_window_plan(
        self,
        window_list: list[Any],
        parallelize_window: bool,
        num_processes: int,
        edge_output_path: str
    ) -> None:
        """
            Print the total number of windows and the selected execution mode.
            :param window_list: [list[Any]] Time-window records returned by TimeWindow.compute_time_windows.
            :param parallelize_window: [bool] Whether windows are processed in parallel.
            :param num_processes: [int] Number of processes used for window execution.
            :param edge_output_path: [str] Directory where temporal edge lists are saved.
            :return: None.
        """
        mode = "parallel" if parallelize_window else "serial"
        self.lm.printl(
            f"[SIM][WINDOW PLAN] total_windows={len(window_list)} execution={mode} processes={num_processes} "
            f"edge_output={edge_output_path} "
            "progress_labels=started_windows/completed_windows"
        )

    def log_window_start(
        self,
        window_index: int,
        n_windows: int,
        start_date: str,
        end_date: str,
        df: pd.DataFrame,
        process_id: int,
        co_action: str
    ) -> None:
        """
            Print a one-line window start summary.
            :param window_index: [int] Start-order window index.
            :param n_windows: [int] Total number of windows in the run.
            :param start_date: [str] Window start date label.
            :param end_date: [str] Window end date label.
            :param df: [pd.DataFrame] Window dataframe.
            :param process_id: [int] Operating-system process id.
            :param co_action: [str] Co-action id currently being processed.
            :return: None.
        """
        self.lm.printl(
            f"[SIM][WINDOW START] started={window_index}/{n_windows} pid={process_id} "
            f"range=[{start_date} -> {end_date}] rows={df.shape[0]} users={df['userId'].nunique()} "
            f"object={action_map[co_action]}"
        )

    def log_window_done(
        self,
        window_index: int,
        completed_index: int,
        n_windows: int,
        start_date: str,
        end_date: str,
        n_edges: int,
        elapsed_seconds: float,
        process_id: int,
        output_path: str | None,
        info_output_path: str | None
    ) -> None:
        """
            Print a one-line window completion summary.
            :param window_index: [int] Start-order window index.
            :param completed_index: [int] Completed-window count.
            :param n_windows: [int] Total number of windows in the run.
            :param start_date: [str] Window start date label.
            :param end_date: [str] Window end date label.
            :param n_edges: [int] Number of edges produced by the window.
            :param elapsed_seconds: [float] Window computation time in seconds.
            :param process_id: [int] Operating-system process id.
            :param output_path: [str | None] Saved edge-list path, or None for empty windows.
            :param info_output_path: [str | None] Saved info-edge-list path, or None when not saved.
            :return: None.
        """
        self.lm.printl(
            f"[SIM][WINDOW DONE] completed={completed_index}/{n_windows} started={window_index}/{n_windows} "
            f"pid={process_id} "
            f"range=[{start_date} -> {end_date}] edges={n_edges} elapsed={elapsed_seconds:.2f}s "
            f"edge_file={output_path if output_path is not None else 'not_saved_empty_window'} "
            f"info_file={info_output_path if info_output_path is not None else 'not_saved'}"
        )

    def log_merge_start(
        self,
        temporal_edge_path: str,
        merged_edge_path: str,
        merge_type: str | None,
        merge_info_edge_list: bool
    ) -> None:
        """
            Print that temporal edge lists are about to be merged.
            :param temporal_edge_path: [str] Source directory containing temporal edge lists.
            :param merged_edge_path: [str] Output directory for the merged edge list.
            :param merge_type: [str | None] Merge strategy configured in TimeWindow.
            :param merge_info_edge_list: [bool] Whether temporal info-edge CSV files will also be merged.
            :return: None.
        """
        self.lm.printl(
            f"[SIM][MERGE START] source={temporal_edge_path} output={merged_edge_path} "
            f"merge={merge_type} merge_info_edge_list={merge_info_edge_list}"
        )

    def log_merge_done(self, merged_edge_path: str) -> None:
        """
            Print that temporal edge-list merging is completed.
            :param merged_edge_path: [str] Output directory containing the merged edge list.
            :return: None.
        """
        self.lm.printl(f"[SIM][MERGE DONE] output={merged_edge_path}")

    def log_run_done(self, dataset_name: str, ca: Any, elapsed_seconds: float) -> None:
        """
            Print the run-level completion summary.
            :param dataset_name: [str] Dataset directory name.
            :param ca: [CoAction] Co-action and similarity-function configuration.
            :param elapsed_seconds: [float] Total compute_similarity runtime in seconds.
            :return: None.
        """
        self.lm.printl(
            f"[SIM][RUN DONE] dataset={dataset_name} co_action={ca.get_co_action()} "
            f"similarity_function={ca.get_similarity_function()} elapsed={elapsed_seconds:.2f}s"
        )
