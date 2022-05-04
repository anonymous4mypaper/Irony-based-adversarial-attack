from gensim.models import KeyedVectors
from util.ltp import *

# Calculate the word mover's distance between two sentences
if __name__ == '__main__':
    s1 = '那个男人真恶心，在公共场所随地吐痰。'
    s2 = '那个男人真优雅，在共场所随地吐痰。真是值得称赞啊。'
    tc_wv_model = KeyedVectors.load_word2vec_format('tencent-ailab-embedding-zh-d100-v0.2.0-s.txt', binary=False)
    distance = tc_wv_model.wmdistance(list(s1), list(segment(s2)), norm=False)
    print(distance)
