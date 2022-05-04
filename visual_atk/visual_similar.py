import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gensim.models import KeyedVectors as W2Vec


def img2ascii(arr):
    result = []
    for row in arr:
        for rgb in row:
            result.append(rgb[0])
    return np.array(result)


def char_img(char):
    width = 24
    height = 24
    image = Image.new('RGB', (width, height))
    font = ImageFont.truetype('visual_atk/simsun.ttf', 22)
    draw = ImageDraw.Draw(image)
    draw.text((2, 0), char, font=font)
    # image.show()
    return np.asarray(image)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def _gen():
    f = open('visual_atk/hanzi.txt', mode='r')
    f2 = open('visual_atk/vce.normalized', mode='a+')
    characters = f.readlines()
    vce = []
    for character in characters:
        arr_char = char_img(character)
        arr_ascii = img2ascii(arr_char)
        arr_ascii = arr_ascii.tolist()
        arr_ascii -= np.mean(arr_ascii)
        vce.append(arr_ascii)
    vce = np.array(vce)
    vce = normalization(vce)
    f2.write(str(len(vce)) + ' ' + str(len(vce[0])) + '\n')
    for i in range(len(vce)):
        s = characters[i].replace('\n', '')
        f2.write(s)
        for j in range(len(vce[i])):
            f2.write(' ' + str(vce[i][j]))
        f2.write('\n')


try:
    model = W2Vec.load_word2vec_format('visual_atk/vce.normalized')
except FileNotFoundError:
    _gen()
    model = W2Vec.load_word2vec_format('visual_atk/vce.normalized')


def similar(char):
    try:
        return model.most_similar(char, topn=10)
    except KeyError:
        return char


def get_vector(char):
    try:
        return model.get_vector(char)
    except KeyError:
        return np.asarray([0.0 for i in range(576)])


if __name__ == '__main__':
    print(random.choice(similar('他')[0]))
