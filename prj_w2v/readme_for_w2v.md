**依赖**
python 版本：3.8
包：
- jieba
- utils
- gensim.models.word2vec
- pandas

**用法**

1. 更改 `json/test.json` 为意图输入的问题
2. 在命令行输入 `python w2v_exec.py`

**各文件用途**

主目录:
- `w2v_wash.py` : 处理数据，将 dataset/ 文件夹中的 14701_q.txt 清洗并分词，结果存在 14701_q_cut.txt 中
- `w2v_train.py`: 训练模型
- `w2v_exec.py` : 初步测试用文件；输入问题，尝试得到回答

`dataset/` 目录:
- `14701_q.txt`: 从原始数据 excel 文件的 Question 栏存储得到的训练集问题
- `14701_q_cut.txt`: 经过分词等处理的训练集问题
- `14701_a.txt`: 从原始数据 excel 文件的 Answer 栏存储得到的训练集回答
- `stopwordslist.txt`: 停用词表
- `usr_dict.txt`: 用户词典

`model/` 目录：
- `w2v.model`: 训练产生的 model 文件

`json/` 目录：
- `test.json`: 测试使用的 json 文件(input)

`test/` 目录：(txt格式的测试数据)
- `1732_test_q.txt`: 从"测试集" excel 文件存储得到的测试集问题
- `22_mini_q.txt`: `1732_test_q.txt` 的子集