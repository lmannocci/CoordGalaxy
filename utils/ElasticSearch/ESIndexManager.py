import json
from utils.LogManager.LogManager import *
# from elasticsearch import Elasticsearch
from utils.ElasticSearch.ESManager import *
from elasticsearch.helpers import bulk
import os
import urllib3
import elasticsearch
urllib3.disable_warnings()

absolute_path = os.path.dirname(__file__)
indeces_path = os.path.join(absolute_path, f"config{os.sep}indeces{os.sep}")
queries_path = os.path.join(absolute_path, f"queries{os.sep}")

class ESIndexManager:
    def __init__(self, es, index_name):
        self.conn = es

        self.index_name = index_name
        self.path_dir_index = indeces_path + self.index_name + "/"
        self.path_file_index = self.path_dir_index + "info.json"
        self.__setEnvIndex()

    def __setEnvIndex(self):
        # create directory Index and last_id_index
        if not os.path.exists(self.path_dir_index):
            os.mkdir(self.path_dir_index)
            os.chmod(self.path_dir_index, 0o777)

        # MANAGE LAST ID INDEX
        # check existing file inside directory
        if not os.path.exists(self.path_file_index):
            self.last_id_index = 0
            data = {"last_id_index": self.last_id_index}
            data_str = json.dumps(data)
            with open(self.path_file_index, 'w') as f:
                f.write(data_str)
        else:
            with open(self.path_file_index, "r", encoding="utf-8") as f:
                c = json.load(f)
                self.last_id_index = c["last_id_index"]

    def existIndex(self):
        return self.conn.es.indices.exists(index=self.index_name)

    def getLastIdIndex(self):
        self.__setEnvIndex()
        if self.existIndex() == True:
            return self.last_id_index
        else:
            self.conn.lm.printl("ESIndexManager. Index " + self.index_name + " does not exist")

    def dataToESFormat(self, data):
        self.__setEnvIndex()
        dataES = []
        for i in range(0, len(data)):
            temp = {}
            temp["_id"] = self.last_id_index + i
            temp["_index"] = self.index_name
            temp["_source"] = data[i]
            dataES.append(temp)
        return dataES

    def uploadData(self, data):
        if self.existIndex() == True:
            data_formatted = self.dataToESFormat(data)
            bulk(self.conn.es, data_formatted)
            self.conn.es.indices.refresh(index=self.index_name)

            self.last_id_index += len(data)
            temp = {"last_id_index": self.last_id_index}
            data_str = json.dumps(temp)
            with open(self.path_file_index, 'w') as f:
                f.write(data_str)

            self.conn.lm.printl("ESIndexManager. Upload in " + self.index_name + " done. Uploaded " + str(len(data)) + " data")
        else:
            self.conn.lm.printl("ESIndexManager. Index " + self.index_name + " does not exist")

    def getCount(self):
        if self.existIndex() == True:
            return self.conn.es.cat.count(index=self.index_name, format="json")
        else:
            self.conn.lm.printl("ESIndexManager. Index " + self.index_name + " does not exist")

    def deleteDocument(self, id_doc):
        if self.existIndex() == True:
            self.conn.es.delete(index=self.index_name, id=id_doc)
        else:
            self.conn.lm.printl("ESIndexManager. Index " + self.index_name + " does not exist")

    def searchQuery(self, query):
        # with open(queries_path + query + ".json", "r", encoding="utf-8") as f:
        #     query = json.load(f)
        self.conn.lm.printl(query)

        if self.existIndex() == True:
            size = 10000
            # Init scroll by search
            if elasticsearch.__version__[0] >= 8:
                data = self.conn.es.search(
                    index=self.index_name,
                    scroll='2m',
                    size=size,
                    query=query,
                    sort={"date": {"order": "asc"}}
                )
            else:
                data = self.conn.es.search(
                    index=self.index_name,
                    scroll='2m',
                    size=size,
                    body=query
                )
            # Get the scroll ID
            sid = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])
            hits = []
            while scroll_size > 0:
                # Before scroll, process current batch of hits
                for hit in data['hits']['hits']:
                    hits.append(hit)
                # process_hits(data['hits']['hits'])
                data = self.conn.es.scroll(scroll_id=sid, scroll='2m')
                # Update the scroll ID
                sid = data['_scroll_id']
                # Get the number of results returned in the last scroll
                scroll_size = len(data['hits']['hits'])
            self.conn.es.clear_scroll(scroll_id=sid)
            result_dict = {'took': 0, 'hits': {'hits': hits}}
            return result_dict
        else:
            self.conn.lm.printl("ESIndexManager. Index " + self.index_name + " does not exist")

    def searchQueryYield(self, query):
        # with open(queries_path + query + ".json", "r", encoding="utf-8") as f:
        #     query = json.load(f)
        self.conn.lm.printl(query)

        if self.existIndex() == True:
            size = 10000
            # Init scroll by search
            if elasticsearch.__version__[0] >= 8:
                data = self.conn.es.search(
                    index=self.index_name,
                    scroll='25m',
                    size=size,
                    query=query,
                    sort={"date": {"order": "asc"}}
                )
            else:
                data = self.conn.es.search(
                    index=self.index_name,
                    scroll='25m',
                    size=size,
                    body=query
                )
            # Get the scroll ID
            sid = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])

            while scroll_size > 0:
                # Before scroll, process current batch of hits
                for hit in data['hits']['hits']:
                    yield hit
                # process_hits(data['hits']['hits'])
                data = self.conn.es.scroll(scroll_id=sid, scroll='25m')
                # Update the scroll ID
                sid = data['_scroll_id']
                # Get the number of results returned in the last scroll
                scroll_size = len(data['hits']['hits'])
            self.conn.es.clear_scroll(scroll_id=sid)
        else:
            self.conn.lm.printl("ESIndexManager. Index " + self.index_name + " does not exist")