import os
from typing import Any

import pandas as pd

from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager
from utils.common_variables import NODE1_VAR, NODE2_VAR, W_VAR, NA_VAR, TW_VAR, dtype, tuple_index
from utils.decorator_definition import log_method


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
data_path = os.path.join(absolute_path, f".{os.sep}..{os.sep}data{os.sep}")
results = os.path.join(absolute_path, f"..{os.sep}results{os.sep}")


class MergeNetworkManager:
    def __init__(
        self,
        dm: Any,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str,
        tw: Any,
        ca: Any
    ) -> None:
        """
            Create the manager that merges temporal edge-list files into one edge list.
            :param dm: [DirectoryManager] Directory manager containing input and output paths.
            :param dataset_name: [str] Dataset directory name.
            :param user_fraction: [float | None] User-selection fraction used in path construction.
            :param type_filter: [str] User-selection strategy name used in path construction.
            :param tw: [TimeWindow] Time-window configuration, including the merge strategy.
            :param ca: [CoAction] Co-action configuration.
            :return: None.
        """
        self.lm = LogManager('main')
        self.dm = dm
        self.ch = Checkpoint()
        self.cm = ConversionManager(dataset_name)

        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.ca = ca

    def _list_files(self, directory_path: str, extension: str) -> list[str]:
        """
            Return sorted files in a directory that match the requested extension.
            :param directory_path: [str] Directory to scan.
            :param extension: [str] File extension including the dot, for example .p or .csv.
            :return: [list[str]] Sorted matching filenames.
        """
        files = sorted(filename for filename in os.listdir(directory_path) if filename.endswith(extension))
        if len(files) == 0:
            message = f"No {extension} files to be merged in {directory_path}."
            self.lm.printl(f"[MERGE][ERROR] {message}")
            raise Exception(message)
        return files

    def _merged_filename(self, temporal_files: list[str]) -> str:
        """
            Build the merged output filename from the first window start and last window end.
            :param temporal_files: [list[str]] Sorted temporal edge-list filenames.
            :return: [str] Merged output filename.
        """
        first_file = temporal_files[0]
        last_file = temporal_files[-1]
        start_date = first_file.split('_')[0]
        end_date = last_file.split('_')[1]
        return f"{start_date}_{end_date}"

    def _edge_list_to_dataframe(self, edge_list: list[tuple]) -> pd.DataFrame:
        """
            Convert one temporal edge list to a dataframe with standard edge columns.
            :param edge_list: [list[tuple]] Temporal edge list loaded from a pickle file.
            :return: [pd.DataFrame] Dataframe containing one row per edge.
        """
        if len(edge_list) == 0:
            return pd.DataFrame()

        max_index = len(edge_list[0])
        columns = list(tuple_index.keys())[0:max_index]
        edge_df = pd.DataFrame(edge_list, columns=columns)
        if NA_VAR not in edge_df.columns:
            edge_df[NA_VAR] = 1
        edge_df[TW_VAR] = 1
        return edge_df.rename(columns={W_VAR: "weightSum"})

    def _append_and_aggregate(self, combined_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
        """
            Append one window dataframe and aggregate repeated edges across windows.
            :param combined_df: [pd.DataFrame] Aggregated dataframe accumulated so far.
            :param edge_df: [pd.DataFrame] Dataframe for the current temporal edge list.
            :return: [pd.DataFrame] Updated aggregated dataframe.
        """
        combined_df = pd.concat([combined_df, edge_df], ignore_index=True)
        return combined_df.groupby([NODE1_VAR, NODE2_VAR]).agg(
            weightSum=("weightSum", "sum"),
            nAction=(NA_VAR, "sum"),
            twCount=(TW_VAR, "sum")
        ).reset_index()

    def _finalize_merged_dataframe(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
            Select the final weight column according to the configured merge strategy.
            :param combined_df: [pd.DataFrame] Aggregated edge dataframe with weightSum and twCount columns.
            :return: [pd.DataFrame] Final dataframe ready to be converted into an edge list.
        """
        if combined_df.empty:
            return pd.DataFrame(columns=[NODE1_VAR, NODE2_VAR, W_VAR, NA_VAR, TW_VAR])

        combined_df["weightAverage"] = combined_df["weightSum"] / combined_df[TW_VAR]
        combined_df[NODE1_VAR] = combined_df[NODE1_VAR].astype(str)
        combined_df[NODE2_VAR] = combined_df[NODE2_VAR].astype(str)

        if self.tw.get_type_merge() == "sum":
            combined_df = combined_df.rename(columns={"weightSum": W_VAR})
            return combined_df[[NODE1_VAR, NODE2_VAR, W_VAR, NA_VAR, TW_VAR]]

        if self.tw.get_type_merge() == "average":
            combined_df = combined_df.rename(columns={"weightAverage": W_VAR})
            return combined_df[[NODE1_VAR, NODE2_VAR, W_VAR, NA_VAR, TW_VAR]]

        message = f"Unsupported merge strategy: {self.tw.get_type_merge()}."
        self.lm.printl(f"[MERGE][ERROR] {message}")
        raise ValueError(message)

    def _merge_edge_dataframes(self, path_edge_list_temporal: str, edge_list_files: list[str]) -> pd.DataFrame:
        """
            Read temporal edge-list files and aggregate them into one dataframe.
            :param path_edge_list_temporal: [str] Directory containing temporal edge-list pickle files.
            :param edge_list_files: [list[str]] Sorted temporal edge-list filenames.
            :return: [pd.DataFrame] Aggregated dataframe before final merge-strategy column selection.
        """
        combined_df = pd.DataFrame()
        skipped_empty_files = 0

        for index, edge_list_file in enumerate(edge_list_files, start=1):
            edge_list_path = f"{path_edge_list_temporal}{edge_list_file}"
            edge_list = self.ch.load_object(edge_list_path)
            if len(edge_list) == 0:
                skipped_empty_files += 1
                self.lm.printl(
                    f"[MERGE][FILE] {index}/{len(edge_list_files)} file={edge_list_file} "
                    "edges=0 status=skipped_empty"
                )
                continue

            edge_df = self._edge_list_to_dataframe(edge_list)
            combined_df = self._append_and_aggregate(combined_df, edge_df)
            self.lm.printl(
                f"[MERGE][FILE] {index}/{len(edge_list_files)} file={edge_list_file} "
                f"edges={len(edge_list)} merged_edges_so_far={combined_df.shape[0]}"
            )

        self.lm.printl(
            f"[MERGE][FILES DONE] total_files={len(edge_list_files)} "
            f"empty_files={skipped_empty_files} aggregated_edges={combined_df.shape[0]}"
        )
        return combined_df

    def _save_merged_edge_list(self, merged_df: pd.DataFrame, output_path: str) -> int:
        """
            Convert the merged dataframe to an edge list and save it.
            :param merged_df: [pd.DataFrame] Final merged edge dataframe.
            :param output_path: [str] Output pickle path.
            :return: [int] Number of merged edges saved.
        """
        merged_edge_list = self.cm.from_df_to_edge_list(merged_df)
        self.ch.save_object(merged_edge_list, output_path)
        return len(merged_edge_list)

    def _merge_info_edge_list(self) -> None:
        """
            Merge temporal info-edge CSV files into one CSV file.
            :return: None. A merged info-edge CSV is written to DirectoryManager.path_info_edge_list.
        """
        info_files = sorted(
            filename for filename in os.listdir(self.dm.path_info_edge_list_temporal) if filename.endswith(".csv")
        )
        if len(info_files) == 0:
            self.lm.printl(
                f"[MERGE][INFO SKIP] no temporal info-edge CSV files in {self.dm.path_info_edge_list_temporal}"
            )
            return

        output_filename = self._merged_filename(info_files)
        output_path = f"{self.dm.path_info_edge_list}{output_filename}.csv"

        self.lm.printl(
            "[MERGE][INFO START] "
            f"files={len(info_files)} source={self.dm.path_info_edge_list_temporal} output={output_path}"
        )
        for index, info_file in enumerate(info_files, start=1):
            info_path = f"{self.dm.path_info_edge_list_temporal}{info_file}"
            info_df = self.ch.read_dataframe(info_path, dtype=dtype)
            info_df.to_csv(output_path, index=False, mode="w" if index == 1 else "a", header=index == 1)
            self.lm.printl(
                f"[MERGE][INFO FILE] {index}/{len(info_files)} file={info_file} rows={info_df.shape[0]}"
            )

        self.lm.printl(f"[MERGE][INFO DONE] output={output_path}")

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    @log_method
    def merge_edge_list(
        self,
        path_edge_list_temporal: str,
        path_edge_list: str,
        merge_info_edge_list: bool = False
    ) -> None:
        """
            Merge temporal edge-list pickle files into one edge list.
            :param path_edge_list_temporal: [str] Directory containing temporal edge-list pickle files.
            :param path_edge_list: [str] Directory where the merged edge-list pickle file is saved.
            :param merge_info_edge_list: [bool] If True, also merge info-edge CSV files. This can produce very large
                files and is disabled by default.
            :return: None. The merged edge-list pickle file is saved to path_edge_list.
        """
        edge_list_files = self._list_files(path_edge_list_temporal, ".p")
        output_filename = self._merged_filename(edge_list_files)
        output_path = f"{path_edge_list}{output_filename}"
        self.lm.printl(
            "[MERGE][START] "
            f"files={len(edge_list_files)} source={path_edge_list_temporal} output={output_path} "
            f"strategy={self.tw.get_type_merge()} merge_info_edge_list={merge_info_edge_list}"
        )

        combined_df = self._merge_edge_dataframes(path_edge_list_temporal, edge_list_files)
        merged_df = self._finalize_merged_dataframe(combined_df)
        n_edges = self._save_merged_edge_list(merged_df, output_path)

        if merge_info_edge_list:
            self._merge_info_edge_list()

        self.lm.printl(
            "[MERGE][DONE] "
            f"output={output_path} edges={n_edges} strategy={self.tw.get_type_merge()}"
        )
