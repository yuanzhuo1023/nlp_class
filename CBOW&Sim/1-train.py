# coding=utf8
'''
训练一个CBOW模型
'''
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from CBOW import CBOW
import jieba
import pickle as pkl
import time
from functools import wraps


def pre_process():
    # 加载数据 sum_list2是训练用的句子 stopwords是停用词
    with open('data/msr_training.utf8', 'r', encoding='utf-8') as f:
        tmp = f.readlines()
    t = []
    for i in tmp:
        t1 = i.replace('“  ', '')
        t2 = t1.replace('\n', '')
        t.append(t2)
    sum_list2 = []
    for i in t:
        sum_2 = i.replace('  ', '')
        sum_list2.append(sum_2)

    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]

    # 去除停用词 构词表
    words_ = []
    vocab = set()
    for sent in sum_list2:
        words = jieba.lcut(sent)  # jieba分词
        temp = []
        for wd in words:
            if wd not in stopwords:
                temp.append(wd)
                vocab.add(wd)
        words_.append(temp)
    vocab_dict = {}
    for i, wd in enumerate(list(vocab)):
        vocab_dict[wd] = i
    return words_, vocab_dict


def train(words, vocab):
    # 构造数据对 设置训练参数
    torch.manual_seed(666)
    vocab_size = len(vocab)
    train_cbow = []
    for line in words:
        for i in range(2, len(line) - 2):
            temp = ([[line[i - 2], line[i - 1]], [line[i + 1], line[i + 2]]], line[i])
            train_cbow.append(temp)
    hidden_dim = 32
    embedding_dim = 64
    losses = []
    model = CBOW(vocab_size, embedding_dim, hidden_dim)  # 模型实例化
    loss_fun = nn.CrossEntropyLoss()
    optimizer = opt.SGD(model.parameters(), lr=3e-2)

    # 开始训练
    for i in tqdm(range(10)):  # 训练10轮
        total_loss = 0
        for context, target in tqdm(train_cbow, position=0):
            context_indexes = []
            for t in context:
                context_indexes.extend([vocab[t[0]], vocab[t[1]]])
            context_indexes = torch.tensor(context_indexes, dtype=torch.long)
            model.zero_grad()
            prob = model(context_indexes, vocab[target])  # 调用模型的forward
            true_label = torch.tensor([vocab[target]], dtype=torch.long)
            loss = loss_fun(prob, true_label)  # 预测概率 真实标签值
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        torch.save(model, 'data/CBOW.bin')


if __name__ == '__main__':
    words_, vocab_ = pre_process()  # 预处理数据
    # 将处理后的数据以pkl格式保存
    with open('data/words.pkl', 'wb') as f:
        pkl.dump(words_, f)
    with open('data/vocab.pkl', 'wb') as f:
        pkl.dump(vocab_, f)

    # 从pkl文件中读取数据
    with open('data/words.pkl', 'rb') as f:
        words_ = pkl.load(f)
    with open('data/vocab.pkl', 'rb') as f:
        vocab_ = pkl.load(f)

    # 训练CBOW模型
    train(words_, vocab_)
