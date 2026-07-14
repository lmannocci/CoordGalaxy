from __future__ import annotations

import json
import os
import shutil
from typing import Any

import urllib3
from elasticsearch import Elasticsearch

from utils.LogManager.LogManager import LogManager


urllib3.disable_warnings()
absolute_path = os.path.dirname(__file__)
config_path = os.path.join(absolute_path, f"config{os.sep}")
indeces_path = os.path.join(absolute_path, f"config{os.sep}indeces{os.sep}")


class ESManager:
    def __init__(self, username: str) -> None:
        """
        Create an Elasticsearch connection from the local credentials file.

        :param username: [str] Credential profile key inside config/credentials.json.
        :return: None.
        """
        self.lm = LogManager("main")
        credentials = self._read_credentials(username)
        self.encripted = credentials["encripted"]
        self.server = credentials["server"]
        self.port = credentials["port"]
        self.url = self._build_url(credentials)
        self.lm.printl("ESManager. " + self.url)
        # self.es = Elasticsearch([self.url], verify_certs=False, timeout=30)
        self.es = Elasticsearch(self.url, timeout=30)

    def _read_credentials(self, username: str) -> dict[str, Any]:
        """
        Read one Elasticsearch credential profile.

        :param username: [str] Credential profile key.
        :return: [dict[str, Any]] Credential dictionary.
        """
        with open(config_path + "credentials.json", "r", encoding="utf-8") as f:
            return json.load(f)[username]

    def _build_url(self, credentials: dict[str, Any]) -> str:
        """
        Build the Elasticsearch URL from credentials.

        :param credentials: [dict[str, Any]] Credential dictionary.
        :return: [str] Elasticsearch URL.
        """
        if credentials["encripted"] is True:
            self.pw = credentials["pw"]
            self.us = credentials["us"]
            return "https://" + self.us + ":" + self.pw + "@" + self.server + ":" + self.port
        return "http://" + self.server + ":" + self.port

    def _index_metadata_paths(self, index_name: str) -> tuple[str, str]:
        """
        Return metadata directory and file paths for an index.

        :param index_name: [str] Elasticsearch index name.
        :return: [tuple[str, str]] Directory path and info.json path.
        """
        dir_path_index = indeces_path + index_name + os.sep
        return dir_path_index, dir_path_index + "info.json"

    def createIndex(self, index_name: str) -> None:
        """
        Create an Elasticsearch index and its local metadata directory when missing.

        :param index_name: [str] Elasticsearch index name.
        :return: None.
        """
        if self.existIndex(index_name) is False:
            # self.lm.printl(config_path + "mappings.json")
            with open(config_path + "mappings.json", "r", encoding="utf-8") as f:
                mappings = json.load(f)
            # print(mappings)
            self.es.indices.create(index=index_name, body=mappings)
            self._ensure_index_metadata(index_name)
            self.lm.printl("ESManager. Index " + index_name + " created")
            return

        self.lm.printl("ESManager. Index " + index_name + " already exist")

    def _ensure_index_metadata(self, index_name: str) -> None:
        """
        Create local metadata files for an index.

        :param index_name: [str] Elasticsearch index name.
        :return: None.
        """
        dir_path_index, file_path = self._index_metadata_paths(index_name)
        # create directory Index
        if not os.path.exists(dir_path_index):
            os.mkdir(dir_path_index)
            os.chmod(dir_path_index, 0o777)

        # creo file dell'ultimo indice di elastic last_id_index
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(json.dumps({"last_id_index": 0}))

    def deleteIndex(self, index_name: str) -> None:
        """
        Delete an Elasticsearch index and its local metadata directory.

        :param index_name: [str] Elasticsearch index name.
        :return: None.
        """
        dir_path_index, _ = self._index_metadata_paths(index_name)
        if self.existIndex(index_name) is True:
            self.es.indices.delete(index=index_name)
            if os.path.exists(dir_path_index):
                # os.remove(dir_path)
                os.chmod(dir_path_index, 0o777)
                shutil.rmtree(dir_path_index, ignore_errors=True)

            self.lm.printl("ESManager. Index " + index_name + " deleted")
            return

        self.lm.printl("ESManager. Index " + index_name + " does not exist")

    def existIndex(self, index_name: str) -> bool:
        """
        Check whether an Elasticsearch index exists.

        :param index_name: [str] Elasticsearch index name.
        :return: [bool] True when the index exists.
        """
        return self.es.indices.exists(index=index_name)
