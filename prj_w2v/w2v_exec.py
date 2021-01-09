#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import utils
from gensim.models import word2vec
import pandas as pd

def delete_r_n(line):
    return line.replace('\r', '').replace('\n', '').strip()

#这里还没改
def jieba_cut(content):
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            word_list.append(word)
    return word_list


def clear_word_not_on_vocab(word_list, vocab):
    new_word_list = []
    for word in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list


# 把一个全是问题的txt读进来，存为list，
# 给list的每一项做分词，
# 然后洗掉不在vocab里的词，
# 仍然返回list
def rcv_vset(dir_vset, vocab):
    vlists = []
    with open(dir_vset, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = delete_r_n(line)
            seg_list = jieba.cut(line)
            vlists.append(clear_word_not_on_vocab(seg_list, vocab))
    return vlists

# 计算相似度
def calc_vlist(vlist, question_list):
    max_score = 0
    index = 0
    i = 0
    for qlist in question_list:
        score = model.n_similarity(qlist, vlist)
        if score > max_score:
            max_score = score
            index = i
        i = i + 1
    return index


if __name__ == '__main__':
    model = word2vec.Word2Vec.load('model\w2v.model')
    vocab = set(model.wv.vocab.keys())
    # vqlist
    vqlist = []
    with open('pdata-txt\in.txt', 'r', encoding='utf-8') as r_vq:
        for line in r_vq:
            vqlist.append(line)
    # rcv
    vlists = rcv_vset('test\\testset.txt',vocab)
    # question_list(washed)
    question_list =[]
    with open('pdata-txt\out.txt', 'r', encoding='utf-8') as r_inq:
        for line in r_inq:
            question_list.append(line.split())
    # answer_list(unwashed)
    answer_list = []
    with open('pdata-txt\ina.txt', 'r', encoding='utf-8') as r_ina:
        for line in r_ina:
            answer_list.append(line)
    # valist
    valist = []
    for vlist in vlists:
        valist.append(answer_list[calc_vlist(vlist,question_list)])
    #print
    j = 0
    for q in vqlist:
        print(q + "\t" + valist[j])
        j = j + 1
