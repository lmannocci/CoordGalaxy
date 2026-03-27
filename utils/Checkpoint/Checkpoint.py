from utils.LogManager.LogManager import *

import pickle
import uunet.multinet as ml
import pandas as pd

# absolute_path = os.path.dirname(__file__)
# results = os.path.join(absolute_path, f"..{os.sep}..{os.sep}results{os.sep}")
file_name = os.path.splitext(os.path.basename(__file__))[0]


class Checkpoint:
    def __init__(self):
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
    def save_object(self, obj, path):
        #path = self.__get_path(filename, dir_path, add_prefix)
        with open(path, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        self.lm.printl(f"{file_name}. saved object: {path}")

    def load_object(self, path):
        # if dir_path == None:
        #     path = results + filename
        # else:
        #     path = dir_path + filename

        with open(path, 'rb') as f:
            obj = pickle.load(f)
        self.lm.printl(f"{file_name}. loaded object: {path}")
        return obj

    def read_dataframe(self, path, dtype, line_terminator=None):
        if line_terminator is None:
            df = pd.read_csv(path, dtype=dtype)
        else:
            df = pd.read_csv(path, dtype=dtype, lineterminator=line_terminator)
        self.lm.printl(f"{file_name}. read_dataframe: {path}")
        return df

    def save_dataframe(self, df, path):
        # path = self.__get_path(filename, dir_path, add_prefix)
        df.to_csv(path, index=False)
        self.lm.printl(f"{file_name}. save_dataframe: {path}")

    def update_dataframe(self, df, path, dtype):
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

    def update_columns_dataframe(self, df, path, join_columns, dtype):
        """
            Reads a dataframe from the given file path, performs an inner join with another dataframe,
            and saves the resulting dataframe to a specified output path.

            Parameters:
            - path (str): Path to the input CSV file to read the dataframe from and to write to.
            - df (pd.DataFrame): The new dataframe to join with.
            - join_columns (list or str): Columns to use for the inner join.

            Returns:
            - pd.DataFrame: The resulting dataframe after the join.
            """
        # Read the existing dataframe from the file
        input_df = pd.read_csv(path, dtype=dtype)

        # Perform the inner join, updating the existing dataframe
        result_df = input_df.merge(df, on=join_columns, how='inner')

        # Save the resulting dataframe
        result_df.to_csv(path, index=False)

        return result_df

    def read_multiplex_network(self, path):
        MG = ml.read(file=path)
        self.lm.printl(f"{file_name}. read_multilayer_network: {path}")
        return MG

    def save_multiplex_network(self, MG, path):
        ml.write(MG, file=path)
        self.lm.printl(f"{file_name}. save_multilayer_network: {path}")


    from datetime import datetime

    def save_txt(self, lines, path, append=False, add_timestamp=False):
        """
        Save a list of strings (or a single string) to a text file, one per line.

        Parameters
        ----------
        lines : list[str] or str
            The text lines to write. If a single string is provided, it is written as one line.
        filepath : str
            Path of the output text file.
        append : bool, optional
            If True, append to the existing file instead of overwriting. Default is False.
        add_timestamp : bool, optional
            If True, adds a timestamp header before the lines. Default is False.
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

        self.lm.printl(f"[✓] Saved {len(lines)} line(s) to {path} ({'append' if append else 'overwrite'} mode).")
