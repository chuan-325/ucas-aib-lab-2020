#!/usr/bin/python
# -*- coding: UTF-8 -*-

import GrobalParament
import jieba
import pandas as pd

# tools
#去掉回车行
def delete_r_n(line):
    return line.replace('\r', '').replace('\n', '').strip()


#读取停用词
def get_stop_words(stop_words_dir):
    stop_words = []
    with open(stop_words_dir, 'r',
              encoding=GrobalParament.encoding) as f_reader:
        for line in f_reader:
            line = delete_r_n(line)
            stop_words.append(line)
    stop_words = set(stop_words)
    return stop_words


#jieba精准分词
def jieba_cut(content, stop_words):
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
    return word_list


#jieba搜索引擎分词
def jieba_cut_for_search(content, stop_words):
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut_for_search(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
    return word_list


#清理不在词汇表中的词语
def clear_word_from_vocab(word_list, vocab):
    new_word_list = []
    for word in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list


#文本预处理第一种方法Pandas
def preprocessing_text_pd(text_dir, after_process_text_dir, stop_word_dir):
    stop_words = get_stop_words(stop_word_dir)
    sentences = []
    df = pd.read_csv(text_dir)
    for index, row in df.iterrows():
        print(index)
        title = delete_r_n(row['title'])  #返回line
        word_list = jieba_cut(title, stop_words)  #返回list
        df.loc[index, 'title'] = ' '.join(word_list)  #返回用空格分隔的line
        sentences.append(word_list)  #列表组成列表
    df.to_csv(after_process_text_dir,
              encoding=GrobalParament.encoding,
              index=False)
    return sentences

# 1 将已经分词完毕的list导入(使用或不使用停用词表)处理导出为csv文件
def txt2csv(txt_dir, out_csv_dir):
    f_reader = open()