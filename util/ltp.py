from pyltp import Parser
from pyltp import Postagger
from pyltp import Segmentor
import os

LTP_DATA_DIR = '/Users/dylan/PycharmProjects/irony/util/ltp_data_v3.4.0'
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
stopwords_path = os.path.join(LTP_DATA_DIR, 'stopwords')
lexicon_path = os.path.join(LTP_DATA_DIR, 'lexicon')
parser = Parser()
segmentor = Segmentor()
postagger = Postagger()
parser.load(par_model_path)
segmentor.load_with_lexicon(cws_model_path, lexicon_path)
postagger.load(pos_model_path)
stopwords = []
with open(stopwords_path) as f:
    lines = f.read()
stopwords = lines.splitlines()


def segment(sentence):
    return segmentor.segment(sentence)


def postag(sentence):
    words = segment(sentence)
    return postagger.postag(words)


def parse(sentence):
    words = segment(sentence)
    postags = postag(sentence)
    return parser.parse(words, postags)


def seg_pos_parse(sentence):
    words = segment(sentence)
    postags = postag(sentence)
    return words, postags, parser.parse(words, postags)


def remove_stopwords(sentence):
    words = segmentor.segment(sentence)
    results = []
    for word in words:
        if word not in stopwords:
            # print(word)
            results.append(word)
    return ''.join(results)
