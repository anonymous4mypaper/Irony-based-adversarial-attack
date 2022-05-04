import pandas as pd
from sklearn.utils import shuffle

from model.classifier import *

if __name__ == '__main__':
    df = pd.read_csv('corpus/Meituan/reviews.csv')
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    df_pos = shuffle(df_pos)
    df_neg = shuffle(df_neg)

    df_test = df_pos[0:500].append(df_neg[0:500])
    df_local = df_pos[500:3500].append(df_neg[500:3500])
    df_victim = df_pos[3500:6500].append(df_neg[3500:6500])
    df_test.to_csv('corpus/Meituan/test.csv', index=None)
    df_local.to_csv('corpus/Meituan/local.csv', index=None)
    df_victim.to_csv('corpus/Meituan/victim.csv', index=None)

    print(df_test)

    classifier = TextCNN()
    classifier.train(x_train=df_local['review'].tolist(), y_train=df_local['label'].tolist())
    classifier.save('model/Meituan/TextCNN/model_local')

    classifier = BidLSTM()
    classifier.train(x_train=df_local['review'].tolist(), y_train=df_local['label'].tolist())
    classifier.save('model/Meituan/BidLSTM/model_local')

    bert = BERT()
    bert.train('corpus/Meituan/local.csv')
    bert.save('model/Meituan/Bert/model_local')

    classifier = TextCNN()
    classifier.train(x_train=df_victim['review'].tolist(), y_train=df_victim['label'].tolist())
    classifier.save('model/Meituan/TextCNN/model_victim')

    classifier = BidLSTM()
    classifier.train(x_train=df_victim['review'].tolist(), y_train=df_victim['label'].tolist())
    classifier.save('model/Meituan/BidLSTM/model_victim')

    classifier = BERT()
    classifier.train('corpus/Meituan/victim.csv')
    classifier.save('model/Meituan/Bert/model_victim')

    df = pd.read_csv('corpus/Amazon/reviews.csv')
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    df_pos = shuffle(df_pos)
    df_neg = shuffle(df_neg)

    df_test = df_pos[0:500].append(df_neg[0:500])
    df_local = df_pos[500:3500].append(df_neg[500:3500])
    df_victim = df_pos[3500:6500].append(df_neg[3500:6500])
    df_test.to_csv('corpus/Amazon/test.csv', index=None)
    df_local.to_csv('corpus/Amazon/local.csv', index=None)
    df_victim.to_csv('corpus/Amazon/victim.csv', index=None)

    classifier = TextCNN()
    classifier.train(x_train=df_local['review'].tolist(), y_train=df_local['label'].tolist())
    classifier.save('model/Amazon/TextCNN/model_local')

    classifier = BidLSTM()
    classifier.train(x_train=df_local['review'].tolist(), y_train=df_local['label'].tolist())
    classifier.save('model/Amazon/BidLSTM/model_local')

    classifier = BERT()
    classifier.train('corpus/Amazon/local.csv')
    classifier.save('model/Amazon/Bert/model_local')

    classifier = TextCNN()
    classifier.train(x_train=df_victim['review'].tolist(), y_train=df_victim['label'].tolist())
    classifier.save('model/Amazon/TextCNN/model_victim')

    classifier = BidLSTM()
    classifier.train(x_train=df_victim['review'].tolist(), y_train=df_victim['label'].tolist())
    classifier.save('model/Amazon/BidLSTM/model_victim')

    classifier = BERT()
    classifier.train('corpus/Amazon/victim.csv')
    classifier.save('model/Amazon/Bert/model_victim')
