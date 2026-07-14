import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd
import uunet.multinet as ml

from utils.LogManager.LogManager import LogManager

# absolute_path = os.path.dirname(__file__)
# results = os.path.join(absolute_path, f"..{os.sep}..{os.sep}results{os.sep}")
file_name = os.path.splitext(os.path.basename(__file__))[0]


class Checkpoint:
    def __init__(self) -> None:
        """
            Create the checkpoint helper used to read and save framework artifacts.
            :return: None.
        """
        self.lm = LogManager('main')

    # def __get_path(self, filename, dir_path, add_prefix):
    #     if dir_path == None:
    #         if add_prefix == True:
    #             path = results + self.filename + '_' + filename
    #         else:
    #             path = results + filename
    #     else:
    #         if add_prefix == True:
    #             path = dir_path + self.filename + '_' + filename
    #         else:
    #             path = dir_path + filename
    #     return path

    # PUBLIC
    # ------------------------------------------------------------------------------------------------------------------
    #dir_path = None, add_prefix = True
    def save_object(self, obj: Any, path: str) -> None:
        """
            Save a Python object with pickle.
            :param obj: [Any] Object to serialize.
            :param path: [str] Output pickle path.
            :return: None.
        """
        #path = self.__get_path(filename, dir_path, add_prefix)
        with open(path, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        self.lm.printl(f"{file_name}. saved object: {path}")

    def load_object(self, path: str) -> Any:
        """
            Load a Python object from a pickle file.
            :param path: [str] Input pickle path.
            :return: [Any] Deserialized object.
        """
        # if dir_path == None:
        #     path = results + filename
        # else:
        #     path = dir_path + filename

        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.lm.printl(f"{file_name}. loaded object: {path}")
        return obj

    def read_dataframe(
        self,
        path: str,
        dtype: dict[str, Any],
        line_terminator: str | None = None,
    ) -> pd.DataFrame:
        """
            Read a CSV dataframe.
            :param path: [str] Input CSV path.
            :param dtype: [dict[str, Any]] Pandas dtype mapping.
            :param line_terminator: [str | None] Optional custom line terminator.
            :return: [pd.DataFrame] Loaded dataframe.
        """
        if line_terminator is None:
            df = pd.read_csv(path, dtype=dtype)
        else:
            df = pd.read_csv(path, dtype=dtype, lineterminator=line_terminator)
        self.lm.printl(f"{file_name}. read_dataframe: {path}")
        return df

    def save_dataframe(self, df: pd.DataFrame, path: str) -> None:
        """
            Save a dataframe to CSV without the index.
            :param df: [pd.DataFrame] Dataframe to save.
            :param path: [str] Output CSV path.
            :return: None.
        """
        # path = self.__get_path(filename, dir_path, add_prefix)
        df.to_csv(path, index=False)
        self.lm.printl(f"{file_name}. save_dataframe: {path}")

    def update_dataframe(self, df: pd.DataFrame, path: str, dtype: dict[str, Any]) -> None:
        """
            Append a dataframe to an existing CSV, or create the CSV when it does not exist.
            :param df: [pd.DataFrame] New rows to append.
            :param path: [str] CSV path to update.
            :param dtype: [dict[str, Any]] Pandas dtype mapping used when reading an existing CSV.
            :return: None.
        """
        self.lm.printl(f"New dataframe shape: {str(df.shape[0])}")
        # Check if the file exists
        if os.path.exists(path):
            # If the file exists, read it
            existing_df = pd.read_csv(path, dtype=dtype)
            self.lm.printl(f"Existing dataframe shape: {str(existing_df.shape[0])}")
            # Append the new results
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            self.lm.printl(f"Existing dataframe shape: 0 (first time).")
            # If the file does not exist, the updated dataframe is just the result
            updated_df = df
        self.lm.printl(f"Dataframe to write shape: {str(updated_df.shape[0])}")
        updated_df.to_csv(path, index=False)
        self.lm.printl(f"{file_name}. update_dataframe: {path}")

    def update_columns_dataframe(
        self,
        df: pd.DataFrame,
        path: str,
        join_columns: list[str] | str,
        dtype: dict[str, Any],
    ) -> pd.DataFrame:
        """
            Update an existing CSV by keeping only rows that match a new dataframe on join columns.
            :param df: [pd.DataFrame] New dataframe to join with the existing CSV.
            :param path: [str] Existing CSV path to read and overwrite.
            :param join_columns: [list[str] | str] Columns used for the inner join.
            :param dtype: [dict[str, Any]] Pandas dtype mapping used when reading the existing CSV.
            :return: [pd.DataFrame] Joined dataframe saved back to path.
            """
        # Read the existing dataframe from the file
        input_df = pd.read_csv(path, dtype=dtype)

        # Perform the inner join, updating the existing dataframe
        result_df = input_df.merge(df, on=join_columns, how='inner')

        # Save the resulting dataframe
        result_df.to_csv(path, index=False)

        return result_df

    def read_multiplex_network(self, path: str) -> Any:
        """
            Read a uunet multiplex network from its text format.
            :param path: [str] Input multiplex graph path, usually multiplex_graph.txt.
            :return: [Any] In-memory uunet multiplex network.
        """
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        self.lm.printl(
            f"{file_name}. read_multiplex_network start: path={path}, "
            f"size={file_size_mb:.2f} MB. Using uunet ml.read; large files can take a long time."
        )
        MG = ml.read(file=path)
        layers = ml.layers(MG)
        self.lm.printl(
            f"{file_name}. read_multiplex_network completed: "
            f"layers={len(layers)}, names={layers}, path={path}"
        )
        return MG

    def save_multiplex_network(self, MG: Any, path: str) -> None:
        """
            Save a uunet multiplex network to its text format.
            :param MG: [Any] In-memory uunet multiplex network.
            :param path: [str] Output multiplex graph path.
            :return: None.
        """
        ml.write(MG, file=path)
        self.lm.printl(f"{file_name}. save_multilayer_network: {path}")


    def save_txt(
        self,
        lines: list[str] | str,
        path: str,
        append: bool = False,
        add_timestamp: bool = False,
    ) -> None:
        """
            Save text lines to a text file.
            :param lines: [list[str] | str] Text lines to write. A single string is written as one line.
            :param path: [str] Output text path.
            :param append: [bool] If True, append to the existing file instead of overwriting.
            :param add_timestamp: [bool] If True, write a timestamp header before the lines.
            :return: None.
        """
        # Ensure input is a list
        if isinstance(lines, str):
            lines = [lines]

        mode = "a" if append else "w"

        with open(path, mode, encoding="utf-8") as f:
            if add_timestamp:
                f.write(f"--- Results at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write("\n".join(lines))
            f.write("\n")  # final newline for safety

        self.lm.printl(f"{file_name}. Saved {len(lines)} line(s) to {path} ({'append' if append else 'overwrite'} mode).")
