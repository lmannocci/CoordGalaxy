from utils.Checkpoint.Checkpoint import Checkpoint
from utils.ConversionManager.ConversionManager import ConversionManager
from utils.LogManager.LogManager import LogManager

import glob
import os


class EdgeListManager:
    def __init__(self) -> None:
        """
            Create a helper for edge-list file transformations.
            :return: None.
        """
        self.ch = Checkpoint()
        self.cm = ConversionManager()
        self.lm = LogManager("main")

    def convert_ids_directory(self, directory_path: str) -> None:
        """
            Convert userId1 and userId2 in all pickle edge-list files under a directory to simple framework ids.
            :param directory_path: [str] Directory containing pickle edge-list files.
            :return: None. Edge-list pickle files are overwritten in place.
        """
        pickle_files = glob.glob(os.path.join(directory_path, "*.p"))
        self.lm.printl(f"Found {len(pickle_files)} pickle files.")

        for index, file_path in enumerate(pickle_files, start=1):
            self.lm.printl(f"[{index}/{len(pickle_files)}] Processing: {os.path.basename(file_path)}")
            edge_list = self.ch.load_object(file_path)
            updated_edge_list = self.cm.user_id_mapper.convert_edge_list(edge_list, direction="to_simple")
            self.ch.save_object(updated_edge_list, file_path)

    def replace_dash_in_edge_list(self, edge_list: list[tuple]) -> list[tuple]:
        """
            Replace '-' with '_' in the first two elements of each edge tuple.
            :param edge_list: [list[tuple]] Edge list where the first two tuple elements are user ids.
            :return: [list[tuple]] Edge list with normalized string user ids.
        """
        return [
            (
                str(userId1).replace("-", "_"),
                str(userId2).replace("-", "_"),
                *rest
            )
            for userId1, userId2, *rest in edge_list
        ]
