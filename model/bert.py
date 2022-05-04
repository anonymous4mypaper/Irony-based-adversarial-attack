import codecs

import pandas as pd
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

config_path = 'util/chinese_rbt6_L-6_H-768_A-12/bert_config_rbt6.json'
checkpoint_path = 'util/chinese_rbt6_L-6_H-768_A-12/bert_model.ckpt'
dict_path = 'util/chinese_rbt6_L-6_H-768_A-12/vocab.txt'

maxlen = 100


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R


token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=128, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


class Bert:
    def __init__(self):
        self.__model = None
        self.__classes_num = 2

    def save(self, path='model/Bert'):
        if self.__model is None:
            raise Exception('model is not trained')
        if not os.path.exists(path):
            os.mkdir(path)
        self.__model.save(path + '/bert.hdf5')

    def load_model(self, path='model/Bert/bert.hdf5'):
        self.__model = self.__model_build()
        self.__model.load_weights(path)

    def __model_build(self):
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        p = Dense(self.__classes_num, activation='softmax')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(1e-5),  # 用足够小的学习率
                      metrics=['accuracy'])
        return model

    def train(self, csv_path, epoch=5):
        train_df = pd.read_csv(csv_path, sep=',', names=['label', 'review'], header=0).astype(str)

        DATA_LIST = []
        for data_row in train_df.iloc[:].itertuples():
            DATA_LIST.append((data_row.review, to_categorical(data_row.label, 2)))
        DATA_LIST = np.array(DATA_LIST)

        self.__model = self.__model_build()

        train_D = data_generator(DATA_LIST, shuffle=True)
        self.__model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=epoch
        )

    def predict_classes(self, sentences):
        test_df = pd.DataFrame({'review': sentences})
        DATA_LIST_TEST = []
        for data_row in test_df.iloc[:].itertuples():
            DATA_LIST_TEST.append((data_row.review, to_categorical(0, 2)))
        DATA_LIST_TEST = np.array(DATA_LIST_TEST)
        test_D = data_generator(DATA_LIST_TEST, shuffle=False)
        pred = self.__model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=0)
        return [np.argmax(x) for x in pred]

    def predict(self, sentences):
        test_df = pd.DataFrame({'review': sentences})
        DATA_LIST_TEST = []
        for data_row in test_df.iloc[:].itertuples():
            DATA_LIST_TEST.append((data_row.review, to_categorical(0, 2)))
        DATA_LIST_TEST = np.array(DATA_LIST_TEST)
        test_D = data_generator(DATA_LIST_TEST, shuffle=False)
        pred = self.__model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=0)
        return pred
