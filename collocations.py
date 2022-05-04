import json
import os


class Collocations:
    def __init__(self):
        self._collocations = {}

    @property
    def collocations(self):
        return self._collocations

    def add(self, noun, adj, label):
        if noun not in self._collocations.keys():
            self.__add_new_noun(noun, adj, label)
        else:
            self.__add_exist_noun(noun, adj, label)

    def __add_new_noun(self, noun, adj, label):
        polarity = {}
        if label:
            polarity['positive'] = 1
            polarity['negative'] = 0
        else:
            polarity['positive'] = 0
            polarity['negative'] = 1
        adjectives = {adj: polarity}
        self._collocations[noun] = adjectives

    def __add_exist_noun(self, noun, adj, label):
        adjectives = self._collocations[noun]
        if adj not in adjectives.keys():
            self.__add_new_adj(noun, adj, label)
        else:
            self.__add_exist_adj(noun, adj, label)

    def __add_new_adj(self, noun, adj, label):
        adjectives = self._collocations[noun]
        polarity = {}
        if label:
            polarity['positive'] = 1
            polarity['negative'] = 0
        else:
            polarity['positive'] = 0
            polarity['negative'] = 1
        adjectives[adj] = polarity

    def __add_exist_adj(self, noun, adj, label):
        adjectives = self._collocations[noun]
        polarity = adjectives[adj]
        if label:
            polarity['positive'] = polarity['positive'] + 1
        else:
            polarity['negative'] = polarity['negative'] + 1
        adjectives[adj] = polarity

    def to_json(self):
        return json.dumps(self._collocations, ensure_ascii=False)

    def from_json(self, json_path):
        if not os.path.exists(json_path):
            f = open(json_path, mode='w', encoding='utf-8')
            f.close()
        if os.path.getsize(json_path) > 0:
            with open(json_path, mode='r', encoding='utf-8') as f:
                obj = json.load(f)
        else:
            obj = {}
        self._collocations = obj
        return obj


