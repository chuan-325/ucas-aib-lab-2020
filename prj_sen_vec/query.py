import time
import json
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# 准备问答库
def prepare_data():
    # 合并两个标注数据集的问答对
    data1 = pd.read_excel('data/标注数据1.xlsx', usecols=[0, 1])
    data1.columns = ['question', 'answer']
    data2 = pd.read_excel('data/标注数据2.xlsx', usecols=[0, 1])
    data2.columns = ['question', 'answer']
    data = pd.concat([data1, data2], sort=True)
    print(data.shape)

    # 分词
    questions = list(data.loc[:, 'question'])
    corpus = []
    for q in questions:
        corpus.append(' '.join(jieba.lcut(str(q))))

    # 分词后的问题转换成词向量
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=['冬奥会', '冬季', '奥林匹克', '运动会'])
    tfidf = vec.fit_transform(corpus).toarray()
    print(tfidf.shape)
    words = vec.get_feature_names()
    with open('data/word.txt', 'w', encoding='utf-8') as f:
        for i in range(len(words)):
            f.write(words[i]+'\n')

    tfidf = list(tfidf)  # 问答库tf-idf矩阵
    words = list(words)  # 特征词列表
    data = data[['question', 'answer']].values  # 问答对集合
    return tfidf, words, data


# 找到用户query中关键词对应的id，以便之后tf-idf数值的比较
def process_query(qr, word_ls):
    split_text = jieba.lcut(qr)
    ids = []
    for word in split_text:
        if word in word_ls:
            ids.append(word_ls.index(word))
    return ids


# 基于tf-idf的数值返回最好的结果
def get_top_ans(ids, tfidf, qr, qa_set):
    year_user = re.findall(r'[1][9]\d{2}|[2][0]\d{2}', str(qr))

    index = 0
    max_row_index = 0
    max_val = 0
    for row in tfidf:
        year_lib = re.findall(r'[1][9]\d{2}|[2][0]\d{2}', str(qa_set[index][0]))
        if len(year_user) == 0 or len(year_lib) == 0 or len(list(set(year_user) & set(year_lib))) != 0:  #
            # 匹配用户query和问题库的年份
            temp = 0
            for i in ids:
                temp += row[i]
            if temp > max_val:
                max_val = temp
                max_row_index = index
        index += 1
    return max_row_index  # 返回最好匹配问答对的index


def json_decode(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        qr = json.load(f)
    print(qr)
    return qr['question']

def json_encode(question, answer):
    output = {'question':question, 'answer':answer}
    file = open('json/output.json', 'w', encoding='utf-8')
    json.dump(output, file)
    print(output)


# 以json格式输入的测试
def json_test(tfidf, word_ls, qa_set):
    qr = json_decode('json/input.json')
    ids_qr = process_query(qr, word_ls)
    index = get_top_ans(ids_qr, tfidf, qr, qa_set)
    ans = qa_set[index][1]
    json_encode(str(qr), str(ans))

# 手动输入的测试
def answer_qr(tfidf, word_ls, qa_set):
    while True:
        qr = input("请问有什么要问我的吗？（按'q'退出哦）")
        start = time.perf_counter()
        if qr == 'q':
            print('拜拜～')
            break
        else:
            ids_qr = process_query(qr, word_ls)
            index = get_top_ans(ids_qr, tfidf, qr, qa_set)
            ans = qa_set[index][1]
            json_encode(str(qr), str(ans))  # 以json格式输出
        print('Time used:', time.perf_counter() - start)
    return

# 准备模型
start = time.perf_counter()
tfidf_matx, word_list, data_set = prepare_data()
print('请问有什么要问我的吗？')
print('Time used:', time.perf_counter() - start)

# 开始测试
start = time.perf_counter()
# json输入
json_test(tfidf_matx, word_list, data_set)
print('Time used:', time.perf_counter() - start)
# 手动输入
answer_qr(tfidf_matx, word_list, data_set)
