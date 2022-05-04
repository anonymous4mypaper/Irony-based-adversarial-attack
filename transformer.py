import re

import pandas as pd

import util.console as console
import util.ltp as ltp
from collocations import Collocations
from generator import Generator
from model.classifier import BERT
from model.classifier import BidLSTM
from model.classifier import FastText
from model.classifier import TextCNN
from ngram import Ngram


# transform the original sentence to irony
class Transformer:

    def __init__(self, ngram_path, collocation_path, dataset, target='BidLSTM'):
        self._ngram = Ngram(2)
        self._ngram.load(ngram_path)
        self._collocation_path = collocation_path
        if target == 'BidLSTM':
            self._classifier = BidLSTM()
            self._classifier.load_model('model/' + dataset + '/BidLSTM/model_local')
        elif target == 'TextCNN':
            self._classifier = TextCNN()
            self._classifier.load_model('model/' + dataset + '/TextCNN/model_local')
        elif target == 'Fasttext':
            self._classifier = FastText()
            self._classifier.load_model('model/' + dataset + '/Fasttext/model_local.bin')
        elif target == 'Bert':
            self._classifier = BERT()
            self._classifier.load_model('model/' + dataset + '/Bert/model_local/bert.hdf5')
        else:
            raise Exception('unknown model')
        gen = Generator()
        self._comments = gen.gen()

    # 句子切分
    @staticmethod
    def __split(sentence):
        return [x for x in re.split('[，,.。！!]', sentence) if x]

    @staticmethod
    def remove_last_punctuation(words):
        last_char = words[len(words) - 1]
        if last_char in '，。？！~,.`':
            del words[len(words) - 1]
            return Transformer.remove_last_punctuation(words)
        else:
            return words

    # 是否为评价句
    def __is_emotional(self, sentence):
        flag = False
        postags = ltp.postag(sentence)
        arcs = ltp.parse(sentence)
        index_n, index_adj = self.__emotional_words_locate(postags, arcs)
        if index_n is not -1:
            flag = True
        return flag, index_n, index_adj

    @staticmethod
    def __emotional_words_locate(postags, arcs):
        for i in range(0, len(arcs)):
            if arcs[i].relation == 'SBV' and postags[i] == 'n' and postags[arcs[i].head - 1] == 'a':
                return i, arcs[i].head - 1
        return -1, -1

    def __append(self, words, times=0):
        irony = ''
        for comment in self._comments:
            irony = ''.join(words) + '，' + comment
            results = self._classifier.predict_classes([irony])
            if results[0] == 1:
                return comment
        if times == 10:
            return ''
        return self.__append(irony, times=times + 1)

    def __switch(self, sentence):
        sentence_seg = self.__split(sentence)
        results = []
        for i in range(0, len(sentence_seg)):
            is_emotion, index_n, index_adj = self.__is_emotional(sentence_seg[i])
            if is_emotion:
                results.append([i, index_n, index_adj])
        if len(sentence_seg) is 1:
            if len(results) is 1:
                return 'single_eval'
            else:
                return 'single_desc'
        elif len(sentence_seg) is 2:
            if len(results) is 1:
                return 'eval_desc'
            elif len(results) is 2:
                return 'double_eval'
            else:
                return 'double_desc'
        else:
            return 'multi'

    def transform(self, sentence):
        __type = self.__switch(sentence)
        index_n = -1
        index_adj = -1
        emotion_index = -1
        sentence_seg = self.__split(sentence)
        emotion_seg = None
        for i in range(0, len(sentence_seg)):
            is_emotion, index_n, index_adj = self.__is_emotional(sentence_seg[i])
            if is_emotion:
                emotion_index = i
                emotion_seg = ltp.segment(sentence_seg[emotion_index])
                break
        words, postags, arcs = ltp.seg_pos_parse(sentence)
        c = Collocations()
        collocations = c.from_json(self._collocation_path)
        if __type is 'single_eval':
            words = self.remove_last_punctuation(words)
            words.append('。')
            return ''.join(words) + self.__append(words)
        elif __type is 'single_desc':
            words = self.remove_last_punctuation(words)
            words.append('。')
            return ''.join(words) + self.__append(words)
        elif __type is 'eval_desc':
            if emotion_seg[index_n] in collocations.keys():
                adjectives = collocations[emotion_seg[index_n]]
                positives = []
                for adj in adjectives.keys():
                    positive = adjectives[adj]['positive']
                    negative = adjectives[adj]['negative']
                    if positive > negative and (positive - negative) > 10:
                        positives.append(adj)
                if len(positives) == 0:
                    emotion_seg[index_adj] = '不错'
                    emotion_seg = ''.join(emotion_seg)
                    sentence_seg[emotion_index] = emotion_seg
                    return '，'.join(sentence_seg) + '。' + self.__append(words)
                sentences = []
                for i in range(0, len(positives)):
                    emotion_seg[index_adj] = positives[i]
                    sentence_seg[emotion_index] = ''.join(emotion_seg)
                    sentences.append('，'.join(sentence_seg))
                score_max = -1
                score_max_index = -1
                for i in range(0, len(sentences)):
                    score = self._ngram.score(sentences[i])
                    if score > score_max:
                        score_max = score
                        score_max_index = i
                return sentences[score_max_index] + self.__append(words)
            if emotion_seg[index_n] not in collocations.keys():
                emotion_seg[index_adj] = '不错'
                emotion_seg = ''.join(emotion_seg)
                sentence_seg[emotion_index] = emotion_seg
                return '，'.join(sentence_seg) + '。' + self.__append(words)
        else:
            words = self.remove_last_punctuation(words)
            words.append('。')
            return ''.join(words) + self.__append(words)


def trans(dataset='Amazon', local_model='Fasttext'):
    output_file = 'corpus/' + dataset + '/irony_' + local_model + '.txt'
    transformer = Transformer('ngram.json', 'collocations.json', dataset, local_model)
    df = pd.read_csv('corpus/' + dataset + '/test.csv')
    reviews = df[df['label'] == 0]['review']
    file = open(output_file, mode='a+')
    total = len(reviews)
    count = 0
    console.msg_time('starting...')
    for review in reviews:
        console.progress(count + 1, total)
        s = transformer.transform(review)
        file.write(s + '\n')
        file.flush()
        count += 1
    console.msg_time('finished')
    file.close()


if __name__ == '__main__':
    trans('Meituan', 'TextCNN')
    trans('Meituan', 'BidLSTM')
    trans('Meituan', 'Bert')
    trans('Amazon', 'TextCNN')
    trans('Amazon', 'BidLSTM')
    trans('Amazon', 'Bert')
