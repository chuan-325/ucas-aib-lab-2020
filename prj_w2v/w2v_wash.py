import jieba
import re

# 读出停用词表
def stopwordslist():
    stopwords = [
        line.strip()
        for line in open('dataset\\stopwordslist.txt', encoding='UTF-8').readlines()
    ]
    return stopwords


stop_list = stopwordslist()

# 分词
def cutword(seg):
    dict = []
    words = jieba.cut(seg)
    for word in words:
        if word not in stop_list:
            dict.append(word)
    string = ' '.join(dict)
    acquire = '[\s\u4e00-\u9fa5A-Za-z0-9]+' # 只保留空格/中文/字母/数字
    result = re.findall(acquire, string)
    return ' '.join(result)

# IO
def clean_up(input_path, output_path):
    to_write = open(output_path, 'w', encoding='UTF-8')
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            to_write.write(' '.join(cutword(line).split()) + '\n')
    to_write.close()


if __name__ == '__main__':
    jieba.load_userdict('dataset\\usr_dict.txt')
    clean_up('dataset\\14701_q.txt', 'dataset\\14701_q_cut.txt')
