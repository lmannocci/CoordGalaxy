from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.common_variables import dtype


@dataclass(frozen=True)
class DatasetPaths:
    """
    Standard dataset paths used by external main scripts.
    """
    dataset_data: str
    original_data: str
    temp_data: str
    co_action_data: str


def build_paths(dataset_name: str, base_dir: str | None = None) -> DatasetPaths:
    """
    Build standard data paths for a dataset main script.

    :param dataset_name: [str] Dataset name, for example moltbook.
    :param base_dir: [str | None] Project root directory. None uses the parent of this utils directory.
    :return: [DatasetPaths] Paths to dataset-local original, temp, and co-action data directories.
    """
    project_root = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(project_root, f"data{os.sep}")
    dataset_path = os.path.join(data_path, f"{dataset_name}{os.sep}")
    return DatasetPaths(
        dataset_data=dataset_path,
        original_data=os.path.join(dataset_path, f"original{os.sep}"),
        temp_data=os.path.join(dataset_path, f"temp_data{os.sep}"),
        co_action_data=os.path.join(dataset_path, f"co_action_data{os.sep}"),
    )


def read_dataset_file(ch: Checkpoint, paths: DatasetPaths, filename: str) -> pd.DataFrame:
    """
    Read a dataset-local source file from data/<dataset>.

    :param ch: [Checkpoint] Checkpoint instance used to read CSV files.
    :param paths: [DatasetPaths] Standard dataset path bundle.
    :param filename: [str] Filename inside data/<dataset>.
    :return: [pd.DataFrame] Loaded dataframe.
    """
    return ch.read_dataframe(f"{paths.dataset_data}{filename}", dtype=dtype)


def read_original_file(ch: Checkpoint, paths: DatasetPaths, filename: str) -> pd.DataFrame:
    """
    Read an original source file from data/<dataset>/original.

    :param ch: [Checkpoint] Checkpoint instance used to read CSV files.
    :param paths: [DatasetPaths] Standard dataset path bundle.
    :param filename: [str] Filename inside data/<dataset>/original.
    :return: [pd.DataFrame] Loaded dataframe.
    """
    return ch.read_dataframe(f"{paths.original_data}{filename}", dtype=dtype)


def read_temp_file(ch: Checkpoint, paths: DatasetPaths, filename: str) -> pd.DataFrame:
    """
    Read a normalized intermediate file from data/<dataset>/temp_data.

    :param ch: [Checkpoint] Checkpoint instance used to read CSV files.
    :param paths: [DatasetPaths] Standard dataset path bundle.
    :param filename: [str] Filename inside data/<dataset>/temp_data.
    :return: [pd.DataFrame] Loaded dataframe.
    """
    return ch.read_dataframe(f"{paths.temp_data}{filename}", dtype=dtype)


def create_directory(file_name: str, path: str) -> None:
    """
    Create a directory when it does not exist and log the operation.

    :param file_name: [str] Name of the caller module, used in log messages.
    :param path: [str] Directory path to create.
    :return: None.
    """
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path, 0o777)
        LogManager("main").printl(f"{file_name}. Created directory {path}")
