#!/usr/bin/python
# -*- coding: UTF-8 -*-
#停用词+分词+数据预处理

import jieba
import pandas as pd

#去掉回车行
def delete_r_n(line):
    return line.replace('\r','').replace('\n','').strip()

#读取停用词
def get_stop_words(stop_words_dir):
    stop_words = []
    with open(stop_words_dir,'r',encoding='utf-8') as f_reader:
        for line in f_reader:
            line = delete_r_n(line)
            stop_words.append(line)
    stop_words = set(stop_words)
    return stop_words

#jieba精准分词
def jieba_cut(content,stop_words):
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
    return word_list

#结巴搜索引擎分词
def jieba_cut_for_search(content,stop_words):
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut_for_search(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
    return word_list

#清理不在词汇表中的词语
def clear_word_from_vocab(word_list,vocab):
    new_word_list = []
    for word  in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list

#文本预处理第一种方法Pandas
def preprocessing_text_pd(text_dir,after_process_text_dir,stop_word_dir):
    stop_words = get_stop_words(stop_word_dir)
    sentences = []
    df = pd.read_csv(text_dir)
    for index,row in df.iterrows():
        print(index)
        title = delete_r_n(row['title'])
        word_list = jieba_cut(title,stop_words)
        df.loc[index,'title'] = ' '.join(word_list)
        sentences.append(word_list)
    df.to_csv(after_process_text_dir,encoding='utf-8',index=False)
    return sentences


if __name__ == "__main__":
    stop_words =get_stop_words('dataset\stopwordslist.txt')
    sentences = preprocessing_text_pd('dataset\14701_q.txt','dataset\14701_q_cut.txt','dataset\stopwordslist.txt')