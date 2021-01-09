#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import json
import time
import utils
from gensim.models import word2vec
import pandas as pd


def delete_r_n(line):
    return line.replace('\r', '').replace('\n', '').strip()


def clear_word_not_on_vocab(word_list, vocab):
    new_word_list = []
    for word in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list


def get_list(dir):
    save_list = []
    with open(dir, 'r', encoding='utf-8') as fr:
        for line in fr:
            save_list.append(line)
    return save_list


def get_lists(dir):
    save_lists = []
    with open(dir, 'r', encoding='utf-8') as fr:
        for line in fr:
            save_lists.append(line.split())
    return save_lists


def get_cut_list(dir_vset, vocab):
    test_q_list = []
    with open(dir_vset, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = delete_r_n(line)
            seg_list = jieba.cut(line)
            test_q_list.append(clear_word_not_on_vocab(seg_list, vocab))
    return test_q_list


def cut_list(becut_list, vocab):
    test_q_list = []
    for item in becut_list:
        item = delete_r_n(item)
        seg_list = jieba.cut(item)
        test_q_list.append(clear_word_not_on_vocab(seg_list, vocab))
    return test_q_list

# 计算单句相似度
def calc_vlist(vlist, train_q_list):
    max_score = 0
    index = 0
    i = 0
    for qlist in train_q_list:
        score = model.n_similarity(qlist, vlist)
        if score > max_score:
            max_score = score
            index = i
        i = i + 1
    return index


# json to list
def json_decode(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    print(qr)
    return qr['question']


# list to json
def json_encode(question, answer):
    output = {'question': question, 'answer': answer}
    file = open('json/output.json', 'w', encoding='utf-8')
    json.dump(output, file)
    print(output)


if __name__ == '__main__':
    # 准备模型
    model = word2vec.Word2Vec.load('model\\w2v.model')
    vocab = set(model.wv.vocab.keys())
    train_q_list = get_lists('dataset\\14701_q_cut.txt')  # train q
    train_a_list = get_list('dataset\\14701_a.txt')
    # 读取问题
    start = time.perf_counter()
    test_q_uc_list = []
    test_q_uc_list.append(json_decode('json\\test.json'))
    #test_q_uc_list = get_list('test\\1732_test_q.txt')
    test_q_list = cut_list(test_q_uc_list, vocab)
    #test_q_list = get_cut_list('test\\1732_test_q.txt', vocab)
    # test a
    test_a_list = []
    j = 0
    for vlist in test_q_list:
        test_a_list.append(train_a_list[calc_vlist(vlist, train_q_list)])
        json_encode(test_q_uc_list[j], test_a_list[j])
        j = j + 1
    print('Time used:', time.perf_counter() - start)