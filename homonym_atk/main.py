import random

import pandas as pd
import pypinyin
from Pinyin2Hanzi import DefaultDagParams, dag

from model.classifier import *
from util.ltp import *


def confidence_var(ori, adv, model, label):
    pred_ori = model.predict([ori])[0]
    pred_adv = model.predict([adv])[0]
    if label == 1:
        return pred_adv[1] - pred_ori[1]
    else:
        return pred_adv[0] - pred_ori[0]


def choice(d, w):
    if len(d) == 0:
        return w
    r = random.choice(d).path[0]
    if r != w:
        return r
    else:
        return choice(d, w)


def replace(word):
    pinyin_list = pypinyin.lazy_pinyin(word)
    d = DefaultDagParams()
    return choice(dag(d, pinyin_list, path_num=5), word)


def important_word(sentence, model):
    score = model.predict([sentence])[0]
    words = list(segment(sentence))
    d = dict()
    for i in range(len(words)):
        tmp = words.copy()
        t = ''
        for j in range(len(words[i])):
            t += '_'
        tmp[i] = t
        score_i = model.predict([''.join(tmp)])[0]
        var = abs(score[0] - score_i[0])
        d[var] = words[i]
    results = []
    for key in sorted(d, reverse=True):
        results.append(d[key])
    return ''.join(results)


def gen(dataset='Amazon', local_model='Fasttext'):
    classifier = None
    if local_model == 'BidLSTM':
        classifier = BidLSTM()
        classifier.load_model('model/' + dataset + '/BidLSTM/model_local')
    elif local_model == 'TextCNN':
        classifier = TextCNN()
        classifier.load_model('model/' + dataset + '/TextCNN/model_local')
    elif local_model == 'Bert':
        classifier = BERT()
        classifier.load_model('model/' + dataset + '/Bert/model_local/bert.hdf5')
    df = pd.read_csv('corpus/' + dataset + '/test.csv')
    reviews = df[df['label'] == 0]['review'].tolist()
    labels = df[df['label'] == 0]['label'].tolist()
    advs = []
    for i in range(len(reviews)):
        words_important = important_word(reviews[i], classifier)
        sentence = reviews[i]
        for j in range(len(words_important)):
            sentence = sentence.replace(words_important[j], replace(words_important[j]))
            pred_label = classifier.predict_classes([sentence])
            if pred_label != labels[i]:
                advs.append(sentence)
                print(sentence)
                break
            elif j == len(words_important) - 1:
                advs.append(sentence)
                print(sentence)
                break
    with open('corpus/' + dataset + '/homonym_' + local_model + '.txt', mode='w+') as f:
        for adv in advs:
            f.write(adv + '\n')


if __name__ == '__main__':
    # gen('Amazon', 'TextCNN')
    # gen('Amazon', 'BidLSTM')
    gen('Amazon', 'Bert')
