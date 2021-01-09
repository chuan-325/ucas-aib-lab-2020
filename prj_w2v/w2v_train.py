#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import utils
from gensim.models import word2vec
import pandas as pd
import sys

def train(dir_txt, dir_model_out):
    buf=[]
    with open(dir_txt, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf.append(list(line.strip('\n').split(' ')))
    print('[Model] Start!')
    model = word2vec.Word2Vec(sentences=buf,
                              size=200,
                              window=5,
                              min_count=1)
    vocab = list(model.wv.vocab.keys())
    model.save(dir_model_out)
    print('[Model] Saved!')

if __name__ == '__main__':
    train('pdata-txt\out.txt', 'model\w2v.model')