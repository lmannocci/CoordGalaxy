import json
from utils.LogManager.LogManager import *
from elasticsearch import Elasticsearch
import os
import shutil
import urllib3

urllib3.disable_warnings()
absolute_path = os.path.dirname(__file__)
config_path = os.path.join(absolute_path, f"config{os.sep}")
indeces_path = os.path.join(absolute_path, f"config{os.sep}indeces{os.sep}")

class ESManager:
    def __init__(self, username):
        self.lm = LogManager('main')
        # self.lm.printl(config_path + "credentials.json")

        with open(config_path + "credentials.json", "r", encoding="utf-8") as f:
            c = json.load(f)[username]
            print(c)
        self.encripted = c['encripted']
        self.server = c["server"]
        self.port = c["port"]
        if self.encripted==True:
            self.pw = c["pw"]
            self.us = c["us"]
            self.url = 'https://' + self.us + ':' + self.pw + '@' + self.server + ':' + self.port
        else:
            self.url = 'http://' + self.server + ':' + self.port
        self.lm.printl("ESManager. " + self.url)
        # self.es = Elasticsearch([self.url], verify_certs=False, timeout=30)
        self.es = Elasticsearch(self.url, timeout=30)

    def createIndex(self, index_name):
        if self.existIndex(index_name) == False:
            # self.lm.printl(config_path + "mappings.json")
            with open(config_path + "mappings.json", "r", encoding="utf-8") as f:
                mappings = json.load(f)
            # print(mappings)
            self.es.indices.create(index=index_name, body=mappings)


            dir_path_index = indeces_path + index_name + os.sep
            file_path = dir_path_index + "info.json"
            # create directory Index
            if not os.path.exists(dir_path_index):
                os.mkdir(dir_path_index)
                os.chmod(dir_path_index, 0o777)

            # creo file dell'ultimo indice di elastic last_id_index
            if not os.path.exists(file_path):
                data = {"last_id_index": 0}
                data_str = json.dumps(data)
                with open(file_path, 'w') as f:
                    f.write(data_str)

            self.lm.printl("ESManager. Index " + index_name + " created")
            return
        else:
            self.lm.printl("ESManager. Index " + index_name + " already exist")
            return

    def deleteIndex(self, index_name):
        dir_path_index = f"{indeces_path}{index_name}{os.sep}"

        file_path = dir_path_index + "info.json"
        if self.existIndex(index_name) == True:
            self.es.indices.delete(index=index_name)
            if os.path.exists(dir_path_index):
                # os.remove(dir_path)
                os.chmod(dir_path_index, 0o777)
                shutil.rmtree(dir_path_index, ignore_errors=True)

            self.lm.printl("ESManager. Index " + index_name + " deleted")
            return
        else:
            self.lm.printl("ESManager. Index " + index_name + " does not exist")

    def existIndex(self, index_name):
        return self.es.indices.exists(index=index_name)
