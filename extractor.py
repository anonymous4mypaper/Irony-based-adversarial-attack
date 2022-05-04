import pandas as pd

import util.console as console
from collocations import Collocations
import util.ltp as ltp
import os


# extract collocations
class Extractor:
    def __init__(self):
        self._collocations = Collocations()

    def __extract(self, label, sentence):
        words, postags, arcs = ltp.seg_pos_parse(sentence)
        nlist = []
        alist = []
        for i in range(0, len(postags)):
            if arcs[i].relation == 'SBV' and postags[i] == 'n' and postags[arcs[i].head - 1] == 'a':
                nlist.append(words[i])
                alist.append(words[arcs[i].head - 1])
            elif arcs[i].relation == 'ATT' and postags[i] == 'a' and postags[arcs[i].head - 1] == 'n':
                nlist.append(words[arcs[i].head - 1])
                alist.append(words[i])
        if len(nlist) is 0:
            return

        for i in range(0, len(nlist)):
            self._collocations.add(nlist[i], alist[i], True if label == 1 else False)

    def extract(self, file_path, json_path, increment=False):
        if increment:
            self._collocations.from_json(json_path)
        df = pd.read_csv(file_path)
        lens_total = len(df.index)
        console.msg_time('scanning corpus...：')
        for index, row in df.iterrows():
            console.progress(index + 1, lens_total)
            self.__extract(row['label'], row['review'])
        collocations = self._collocations.to_json()
        console.msg_time('writing to file...')
        with open(json_path, mode='w') as f:
            f.write(collocations)
        console.msg_time('finished')


if __name__ == '__main__':
    extractor = Extractor()
    for root, dirs, files in os.walk("corpus"):
        for file in files:
            path = os.path.join(root, file)
            if not path.endswith('csv'):
                continue
            print(path)
            extractor.extract(path, 'collocations.json', increment=True)
