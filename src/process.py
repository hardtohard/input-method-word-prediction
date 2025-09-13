import pandas as pd
import config
import tqdm
from sklearn.model_selection import train_test_split
import jieba

def build_dataset(sentences,word2index):
    """
    :param word2index:{word:index}
    :param sentences:list['我爱自然语言',['我不爱自然语言']]
    :return:{input:[1,2,3,4,5],target:6,input:[2,3,4,5,6],target:7}
    """
    index_sentences=[[word2index.get(word,0) for word in jieba.lcut(sentence)] for sentence in sentences]

    #构建一个存储的列表,列表中存储的是字典{input:[1,2,3,4,5],target:6,input:[2,3,4,5,6],target:7}

    train_dataset=[]
    #滑动窗口
    for sentence in index_sentences:
        # sentence[1,2,3,4,5,6,,7,8,9,10]
        for i in range(len(sentence)-config.SEQ_LEN):
            input=sentence[i:i+config.SEQ_LEN]
            target=sentence[i+config.SEQ_LEN]
            train_dataset.append({"input":input,"target":target})
    return train_dataset

def process():
    """
    数据预处理
    """

    print("开始处理数据")
    #数据读取
    df=pd.read_json(config.RAW_DATA_DIR/'synthesized_.jsonl',lines=True, orient='records').sample(frac=0.1)
    print(df.head())
    #抽取句子
    sentences=[]
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])
    #print(f'sentences is {sentences}')
    print(f'句子总数是{len(sentences)}')

    #划分数据集


    train_sentences,test_sentences=train_test_split(sentences,test_size=0.2)
    print(f'训练集句子长度{len(train_sentences)}')
    print(f'测试集句子长度{len(test_sentences)}')

    #构建词表(用训练集)
    vocab_set=set()
    for sentence in tqdm.tqdm(train_sentences,desc="构建词表"):
        for word in jieba.lcut(sentence):
            vocab_set.add(word)
    vocab_list=['<unk>']+list(vocab_set)
    print(f'词表大小:{len(vocab_list)}')
    #print(f'词表长这样：{vocab_list}')

    word2index={word: index for index, word in enumerate(vocab_list)}
    #构建训练集
    train_dataset=build_dataset(train_sentences,word2index)
    #构建测试集
    test_dataset=build_dataset(test_sentences,word2index)
    # 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DIR/'index_train.jsonl',lines=True,orient='records')
    #保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DIR / 'index_test.jsonl', lines=True, orient='records')
    print("数据处理完毕")


if __name__ == '__main__':
    process()