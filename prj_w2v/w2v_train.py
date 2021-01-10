#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import utils
from gensim.models import word2vec
import pandas as pd
import sys

# 使用 gensim 中集成的 word2vec 包，训练模型
def train(dir_txt, dir_model_out):
    # 将原始问句集读入 buf[]
    buf=[]
    with open(dir_txt, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf.append(list(line.strip('\n').split(' ')))

    print('[Model] Start!')
    # 调用 word2vec.Word2Vec 完成 word-embedding
    model = word2vec.Word2Vec(sentences=buf, # 待处理的句子
                              size=200,      # 向量空间的维数
                              window=5,      # 上下文窗口大小(词数)
                              min_count=1)   # 最小统计词频(小于它则忽略)
    vocab = list(model.wv.vocab.keys()) # 将 model 中出现过的所有词存入 vocab
    model.save(dir_model_out) # 保存 model 文件
    print('[Model] Saved!')


if __name__ == '__main__':
    train('dataset\\14701_q_cut.txt', 'model\\w2v.model')