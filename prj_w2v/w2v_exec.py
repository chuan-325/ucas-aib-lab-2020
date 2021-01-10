#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import json
import time
import utils
from gensim.models import word2vec
import pandas as pd

# 删除 \r(return) 和 \n(newline)
#   输入： line
#   输出： 经过处理的 line
def delete_r_n(line):
    return line.replace('\r', '').replace('\n', '').strip()

# 删除分词句内不在 vocabulary 里的 word
#   输入： line, vocab
#   输出： 没有生词的 line
def clear_word_not_on_vocab(word_list, vocab):
    new_word_list = []
    for word in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list

# 读取文件，将每一行作为表项，生成 list
#   输入： 文件目录
#   输出： list
def get_list(dir):
    save_list = []
    with open(dir, 'r', encoding='utf-8') as fr:
        for line in fr:
            save_list.append(line)
    return save_list

# 读取文件，将每一行进行分词，每一行成为一个 list，最后生成'行list' 组成的 list
#   输入： 文件目录
#   输出： 行list 组成的 list
def get_lists(dir):
    save_lists = []
    with open(dir, 'r', encoding='utf-8') as fr:
        for line in fr:
            save_lists.append(line.split()) #这里比上面多出的 split 会按照空格将分词过的行处理为 list
    return save_lists

# 对 list 中的每一项进行分词
#   输入： 句子组成的 list
#   输出： 经过分词且无生词的句子组成的 list
def cut_list(becut_list, vocab):
    test_q_list = []
    for item in becut_list:
        item = delete_r_n(item)
        seg_list = jieba.cut(item)
        test_q_list.append(clear_word_not_on_vocab(seg_list, vocab))
    return test_q_list

# 寻找与输入的问题(单个)相似度最高的训练集问题
#   输入： 句子组成的 list, 训练集问题组成的 list(已分词)
#   输出： 与输入的问题(单个)相似度最高的训练集问题的下标
def calc_vlist(vlist, train_q_list):
    max_score = 0
    index = 0
    i = 0
    for qlist in train_q_list:
        score = model.wv.n_similarity(qlist, vlist)
        if score > max_score:
            max_score = score
            index = i
        i = i + 1
    return index

# 解码 json 输入(@yxy)
#   输入： 文件路径
#   输出： 单个 question 项组成的 list
def json_decode(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    print(qr)
    return qr['question']

# 编码 json 输出(@yxy)
#   输入： 测试问题，找到的答案
#   输出： 无(打印并在文件中写入结果)
def json_encode(question, answer):
    output = {'question': question, 'answer': answer}
    file = open('json\\output.json', 'w', encoding='utf-8')
    json.dump(output, file)
    print(output)

if __name__ == '__main__':
    # 准备模型
    model = word2vec.Word2Vec.load('model\\w2v.model')    # 加载 model 文件
    vocab = set(model.wv.vocab.keys())                    # 将词汇表保存在 vocab 中
    train_q_list = get_lists('dataset\\14701_q_cut.txt')  # 训练集问题读入
    train_a_list = get_list('dataset\\14701_a.txt')       # 训练集答案读入
    # 读取问题
    start = time.perf_counter()
    test_q_uc_list = []
    test_q_uc_list.append(json_decode('json\\test.json')) # 得到输入的测试问题
    test_q_list = cut_list(test_q_uc_list, vocab)         # 做分词和去生词等处理
    # 准备生成答案
    test_a_list = []
    j = 0
    for vlist in test_q_list:
        fit_index = calc_vlist(vlist, train_q_list) # 计算得到[训练集问题]中与[输入]相似度最高项的下标
        fit_answer = train_a_list[fit_index]        # 该下标对应的[答案]
        test_a_list.append(fit_answer)              # 将答案添加到列表中
        json_encode(test_q_uc_list[j], fit_answer)  # 答案输出
        j = j + 1
    print('Time used:', time.perf_counter() - start)
