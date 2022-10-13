# Code and data of the paper "IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems".

## Requirements

- gensim==4.1.2
- Keras==2.4.3
- keras-bert==0.88.0
- numpy==1.19.5
- pandas==1.1.5
- Pinyin2Hanzi==0.1.1
- pyemd==0.5.1
- pyltp==0.2.1
- pypinyin==0.46.0
- tensorflow==2.5.0

## General Required Data and Tools

Download [ltp_data_v3.4.0](https://anonfiles.com/RcA8Ufc0y6/ltp_data_v3.4.0_zip)
&emsp;data for Chinese word segmentation and part of speech tagging and dependency analysis tools

Download [chinese_rbt6_L-6_H-768_A-12](https://anonfiles.com/9462U5c5y2/chinese_rbt6_L-6_H-768_A-12_zip)
&emsp;bert pretrained model

Download [vce.normalized](https://anonfiles.com/r949Uac0yc/vce_normalized)
&emsp;visual embeddings for visual based attack

Download [tencent-ailab-embedding-zh-d100-v0.2.0-s.txt](https://anonfiles.com/50I1Uac5y5/tencent-ailab-embedding-zh-d100-v0.2.0-s_txt)
&emsp;word embeddings for chinese

Download [collocations.json](https://anonfiles.com/XdK6V6c4y5/collocations_json)
&emsp; collocations.json extracted by us

Download [ngram.json](https://anonfiles.com/3eMbV3c0y3/ngram_json)
&emsp; ngram.json extracted by us

Download [simplifyweibo_4_moods.csv](https://anonfiles.com/paL6Vcc4yb/simplifyweibo_4_moods_csv)
&emsp; Sina weibo comments

Download [news.zip](https://anonfiles.com/d6P7V2c3ye/news_zip)
&emsp; News corpus

## Usage

### train models

Run
``shell python train.py
``
for training all local and victim models (TextCNN, BidLSTM, Bert).

### extract collocations

Run
``shell python extractor.py
``
for extracting collocations from Chinese corpus.

### build ngram.json

Run
``shell python ngram.py
``
for building ngram.json from Chinese corpus.

### generate adversarial examples

Run
``shell python transformer.py
``
for generating irony-based adversarial examples for all local models.




