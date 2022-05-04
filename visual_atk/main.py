from model.classifier import *
from util import console
from util.ltp import *
import pandas as pd
import random
from ngram import Ngram
from visual_atk import visual_similar


def confidence_var(ori, adv, model, label):
    """
    置信度变化程度
    """
    pred_ori = model.predict([ori])[0]
    pred_adv = model.predict([adv])[0]
    if label == 1:
        return pred_adv[1] - pred_ori[1]
    else:
        return pred_adv[0] - pred_ori[0]


def replace(word):
    results = []
    for w in word:
        results.append(random.choice(visual_similar.similar(w))[0])
    return ''.join(results)


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


# if __name__ == '__main__':
#     n = Ngram(2)
#     n.load('ngram.json')
#     df = pd.read_csv('corpus/Meituan/test.csv')
#     reviews = df['review']
#     labels = df['label']
#     f = open('corpus/Meituan/visual_BidLSTM.txt')
#     advs = f.readlines()
#
#     ori_perplexity = 0
#     adv_perplexity = 0
#     for review in reviews:
#         perplexity = n.perplexity(review)
#         ori_perplexity += perplexity
#     for adv in advs:
#         perplexity = n.perplexity(adv)
#         adv_perplexity += perplexity
#     print(ori_perplexity / len(reviews))
#     print(adv_perplexity / len(advs))


# if __name__ == '__main__':
#     classifier = BidLSTM()
#     classifier.load_model('model/Meituan/BidLSTM/model_local')
#     df = pd.read_csv('corpus/Meituan/test.csv')
#     reviews = df['review']
#     labels = df['label']
#     f = open('corpus/Meituan/visual_BidLSTM.txt')
#     advs = f.readlines()
#     total = 0
#     for i in range(len(reviews)):
#         review = reviews[i]
#         var = confidence_var(review, advs[i], classifier, labels[i])
#         total += var
#     print(total / len(reviews))

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
    with open('corpus/' + dataset + '/visual_' + local_model + '.txt', mode='w+') as f:
        for adv in advs:
            f.write(adv + '\n')


if __name__ == '__main__':
    gen('Amazon', 'TextCNN')
    gen('Amazon', 'BidLSTM')
    gen('Amazon', 'Bert')
