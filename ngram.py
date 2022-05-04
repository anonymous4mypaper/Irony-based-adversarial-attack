import pandas as pd
import json
import os
import util.console as console
import util.ltp as ltp


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        return {'count': obj.count, 'pre_words': obj.pre_words}


class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict2object)

    def dict2object(self, d):
        # convert dict to object
        if '__class__' in d:
            d.pop('__class__')
            d.pop('__module__')
            args = dict((key, value) for key, value in d.items())  # get args
            inst = Word()  # create new instance
            inst.init_all(**args)
        else:
            inst = d
        return inst


class Word:
    def __init__(self):
        self._count = 0
        self._pre_words = {}

    def init_all(self, count, pre_words):
        self._count = count
        self._pre_words = pre_words

    @property
    def count(self):
        return self._count

    @property
    def pre_words(self):
        return self._pre_words

    def add(self, pre_word):
        if pre_word not in self._pre_words.keys():
            self._pre_words[pre_word] = 1
            self._count = self._count + 1
        else:
            self._pre_words[pre_word] = self._pre_words[pre_word] + 1
            self._count = self._count + 1


class LanguageModel:
    def __init__(self):
        self._table = {}

    @property
    def table(self):
        return self._table

    def add(self, key, pre_word):
        if key in self._table.keys():
            if isinstance(self._table[key], dict):
                word = Word()
                word.init_all(self._table[key]['count'], self._table[key]['pre_words'])
            else:
                word = self._table[key]
        else:
            word = Word()
        word.add(pre_word)
        self._table[key] = word

    def train_txt(self, src):
        try:
            with open(src, encoding='gbk') as txt:
                for line in txt.readlines():
                    words = ltp.segment(ltp.remove_stopwords(line))
                    for i in range(0, len(words)):
                        if i == 0:
                            self.add(words[i], '<s>')
                        else:
                            self.add(words[i], words[i - 1])
        except UnicodeDecodeError as e:
            try:
                with open(src, encoding='utf-8') as txt:
                    for line in txt.readlines():
                        words = ltp.segment(ltp.remove_stopwords(line))
                        for i in range(0, len(words)):
                            if i == 0:
                                self.add(words[i], '<s>')
                            else:
                                self.add(words[i], words[i - 1])
            except UnicodeDecodeError as e:
                pass

    def train_csv(self, src):
        df = pd.read_csv(src)
        lens_total = len(df.index)
        console.msg_time('开始扫描语料库并统计')
        for index, row in df.iterrows():
            console.progress(index + 1, lens_total)
            sentence = row['review']
            words = ltp.segment(sentence)
            for i in range(0, len(words)):
                if i == 0:
                    self.add(words[i], '<s>')
                else:
                    self.add(words[i], words[i - 1])

    def train(self, src, json_path, increment=False):
        if increment:
            self._table = self.from_json(json_path)
        if src.endswith('.txt'):
            self.train_txt(src)
        elif src.endswith('.csv'):
            self.train_csv(src)
        with open(json_path, mode='w') as f:
            f.write(self.to_json())
        console.msg_time('ngram generate/update finished!')

    def to_json(self):
        return json.dumps(self._table, cls=MyEncoder, ensure_ascii=False)

    @staticmethod
    def from_json(json_path):
        if not os.path.exists(json_path):
            f = open(json_path, mode='w', encoding='utf-8')
            f.close()
        if os.path.getsize(json_path) > 0:
            with open(json_path, mode='r', encoding='utf-8') as f:
                json_data = json.dumps(json.load(f))
                obj = MyDecoder().decode(json_data)
        else:
            obj = {}
        return obj


class Ngram:
    def __init__(self, n):
        self._n = n
        self._model = None

    def load(self, json_path):
        print("加载ngram模型文件...")
        self._model = LanguageModel.from_json(json_path)

    def train(self, src, json_path, increment=False):
        model = LanguageModel()
        model.train(src, json_path, increment)
        self._model = model.table

    def score(self, sentence):
        words = ltp.segment(sentence)
        p = 0
        for i in range(0, len(words)):
            if i == 0:
                p = self.probability(words[i], '<s>')
            else:
                p = p * self.probability(words[i], words[i - 1])
        return p

    def perplexity(self, sentence):
        score = self.score(sentence)
        length = len(sentence)
        if score == 0:
            score = 1.0e-5
        return pow(1 / score, 1 / length)

    def probability(self, word, pre_word):
        if word not in self._model:
            return 1.0e-5
        else:
            total = self._model[word]["count"]
            if pre_word in self._model[word]["pre_words"].keys():
                count = self._model[word]['pre_words'][pre_word]
            else:
                return 1.0e-5
        p = count / total
        return p


if __name__ == '__main__':
    ngram = Ngram(2)

    for root, dirs, files in os.walk("corpus"):
        for file in files:
            path = os.path.join(root, file)
            print(path)
            ngram.train(path, 'ngram.json', increment=True)
    ngram.train('corpus/复旦中文文本分类数据集/C3-Art/C3-Art0001.txt', 'test.txt', increment=True)
