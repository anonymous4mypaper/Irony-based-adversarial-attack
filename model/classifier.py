import os
import random as rn

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1205)
rn.seed(1205)
tf.random.set_seed(1205)
import pickle

import fasttext

from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Conv1D, MaxPooling1D, \
    concatenate, \
    Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras import Model

from model.bert import Bert
from util import ltp

max_words = 5000
max_len = 50


class TextCNN:
    def __init__(self):
        self.__model = None
        self.__tok = None
        self.__classes_num = -1

    def save(self, path='model/BidLSTM'):
        if self.__model is None:
            raise Exception('model is not trained')
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + '/tok.pickle', 'wb') as handle:
            pickle.dump(self.__tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.__model.save(path)

    def load_model(self, path='model/BidLSTM'):
        with open(path + '/tok.pickle', 'rb') as handle:
            self.__tok = pickle.load(handle)
        self.__model = load_model(path)

    def __model_build(self):
        main_input = Input(shape=[max_len], dtype='float64')
        embedder = Embedding(max_words + 1, 300, input_length=max_len)
        embed = embedder(main_input)
        cnn1 = Conv1D(512, 3, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=48)(cnn1)
        cnn2 = Conv1D(512, 4, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=47)(cnn2)
        cnn3 = Conv1D(512, 5, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=46)(cnn3)
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(0.5)(flat)
        main_output = Dense(self.__classes_num, activation='sigmoid')(drop)
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, batch_size=128, epoch=2):
        x_train, y_train = self.__pre_process(x_train, y_train)
        self.__model = self.__model_build()
        self.__model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch)

    def evaluate(self, x_test, y_test):
        x_test, y_test = self.__pre_process(x_test, y_test)
        return self.__model.evaluate(x_test, y_test, verbose=0)

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=-1)

    def predict(self, X):
        if not isinstance(X, list):
            raise Exception('Only accept `list` of `str`')
        x = self.__pre_process(X)
        return self.__model.predict(x)

    def __pre_process(self, X, y=None):
        sentences = []
        for x in X:
            words = ltp.segment(x)
            sentence = ''
            for word in words:
                sentence = sentence + word + ' '
            sentences.append(sentence)

        if y is not None:
            le = LabelEncoder()
            y = le.fit_transform(y).reshape(-1, 1)
            ohe = OneHotEncoder()
            y = ohe.fit_transform(y).toarray()
            self.__classes_num = y.shape[1]

        if self.__tok is None:
            self.__tok = Tokenizer(num_words=max_words, filters='！，。!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
            self.__tok.fit_on_texts(sentences)
        seq = self.__tok.texts_to_sequences(sentences)
        seq_mat = sequence.pad_sequences(seq, maxlen=max_len)
        if y is None:
            return seq_mat
        return seq_mat, y


class BidLSTM:
    def __init__(self):
        self.__model = None
        self.__tok = None
        self.__classes_num = -1

    def save(self, path='model/BidLSTM'):
        if self.__model is None:
            raise Exception('model is not trained')
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + '/tok.pickle', 'wb') as handle:
            pickle.dump(self.__tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.__model.save(path)

    def load_model(self, path='model/BidLSTM'):
        with open(path + '/tok.pickle', 'rb') as handle:
            self.__tok = pickle.load(handle)
        self.__model = load_model(path)

    def __model_build(self):
        model = Sequential()
        model.add(Input(name='inputs', shape=[max_len]))
        model.add(Embedding(max_words + 1, 300, input_length=max_len))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.5))
        model.add(Dense(self.__classes_num, activation='sigmoid'))
        model.compile(Adam(0.01), 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, batch_size=128, epoch=2):
        x_train, y_train = self.__pre_process(x_train, y_train)
        self.__model = self.__model_build()
        self.__model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch)

    def evaluate(self, x_test, y_test):
        x_test, y_test = self.__pre_process(x_test, y_test)
        return self.__model.evaluate(x_test, y_test, verbose=0)

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=-1)

    def predict(self, X):
        if not isinstance(X, list):
            raise Exception('Only accept `list` of `str`')
        x = self.__pre_process(X)
        return self.__model.predict(x)

    def __pre_process(self, X, y=None):
        sentences = []
        for x in X:
            words = ltp.segment(x)
            sentence = ''
            for word in words:
                sentence = sentence + word + ' '
            sentences.append(sentence)

        if y is not None:
            le = LabelEncoder()
            y = le.fit_transform(y).reshape(-1, 1)
            ohe = OneHotEncoder()
            y = ohe.fit_transform(y).toarray()
            self.__classes_num = y.shape[1]

        if self.__tok is None:
            self.__tok = Tokenizer(num_words=max_words, filters='！，。!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
            self.__tok.fit_on_texts(sentences)
        seq = self.__tok.texts_to_sequences(sentences)
        seq_mat = sequence.pad_sequences(seq, maxlen=max_len)
        if y is None:
            return seq_mat
        return seq_mat, y


class FastText:

    def __init__(self):
        self.__model = None

    def __pre_process(self, X, y=None):
        f = open('corpus/waimai_10k.txt', mode='w+')
        for i in range(len(X)):
            words = ltp.segment(X[i])
            sentence = ''
            for word in words:
                sentence += word + ' '
            f.write(sentence + '\t__label__' + str(y[i]) + '\n')
        f.close()

    def train(self, x_train, y_train, epoch=50):
        self.__pre_process(x_train, y_train)
        self.__model = fasttext.train_supervised(
            input='corpus/waimai_10k.txt',
            label_prefix='__label__',
            dim=256,
            epoch=epoch,
            lr=0.01,
            min_count=3,
            loss='softmax',
            word_ngrams=2,
            bucket=1000000
        )
        os.remove('corpus/waimai_10k.txt')

    def save(self, path='model/Fasttext/Model.bin'):
        self.__model.save_model(path)

    def load_model(self, path='model/Fasttext/Model.bin'):
        self.__model = fasttext.load_model(path)

    def predict_classes(self, X):
        if not isinstance(X, list):
            raise Exception('Only accept `list` of `str`')
        sentences = []
        for x in X:
            sentence = ''
            words = ltp.segment(x)
            for word in words:
                sentence += word + ' '
            sentences.append(sentence)
        pred = self.__model.predict(sentences)
        result = []
        for i in range(len(pred[0])):
            if pred[0][i][0] == '__label__1':
                result.append(1)
            elif pred[0][i][0] == '__label__0':
                result.append(0)
        return result


class BERT:
    def __init__(self):
        self.__bert = Bert()

    def train(self, csv_path, epoch=2):
        self.__bert.train(csv_path, epoch)

    def save(self, path='model/Bert'):
        self.__bert.save(path)

    def load_model(self, path='model/Bert/bert.hdf5'):
        self.__bert.load_model(path)

    def predict_classes(self, X):
        return self.__bert.predict_classes(X)

    def predict(self, X):
        return self.__bert.predict(X)
