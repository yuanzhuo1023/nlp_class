"""中文分词
步骤二:
训练分词模型
    1-模型参数 Config
    2-前馈网络模型 Model
    3-评估模型 model_eval

    4：训练及测试过程: main
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.vocab = pkl.load(open('./vocab.pkl', 'rb'))  # 读取词表
        self.train_data = pkl.load(open('./train_data.pkl', 'rb'))  # 读取训练数据
        self.target = pkl.load(open('./target.pkl', 'rb'))  # 读取标签

        self.learning_rate = 0.00015  # 学习率
        self.epoch = 50  # epoch次数

        self.output_size = 4
        self.embed_dim = 32


class Model(nn.Module):
    """
    结构: input->嵌入层->隐藏层->输出层->output
    """
    def __init__(self, output_size, vocab_size, embed_dim):
        super(Model, self).__init__()
        self.hid_layer = nn.Linear(embed_dim, 64)  # 隐藏层
        self.out_layer = nn.Linear(64, output_size)  # 输出层
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 嵌入层

    def forward(self, in_layer):
        emd = self.embedding(in_layer)
        h_out = self.hid_layer(emd)
        h_out = F.relu(h_out)
        out_ = self.out_layer(h_out)
        out_ = F.softmax(out_, dim=1)
        return out_


def model_eval(model_out, true_label):
    confusion_matrix = torch.zeros([2, 2], dtype=torch.long)
    predict_label = torch.argmax(model_out, 1)
    accuracy = []
    precision = []
    recall = []
    f_1 = []
    for l in range(4):
        tp_num, fp_num, fn_num, tn_num = 0, 0, 0, 0
        for p, t in zip(predict_label, true_label):
            if p == t and t == l:
                tp_num += 1
            if p == l and t != l:
                fp_num += 1
            if p != l and p != t:
                fn_num += 1
            if p != l and p == t:
                tn_num += 1
        accuracy.append((tp_num + tn_num) / (tp_num + tn_num + fp_num + fn_num))
        try:
            prec = tp_num / (tp_num + fp_num)
        except:
            prec = 0.0
        try:
            rec = tp_num / (tp_num + fn_num)
        except:
            rec = 0
        precision.append(prec)
        recall.append(rec)
        if prec == 0 and rec == 0:
            f_1.append(0)
        else:
            f_1.append((2 * prec * rec) / (prec + rec))
    ave_acc = torch.tensor(accuracy, dtype=torch.float).mean()
    ave_prec = torch.tensor(precision, dtype=torch.float).mean()
    ave_rec = torch.tensor(recall, dtype=torch.float).mean()
    ave_f1 = torch.tensor(f_1, dtype=torch.float).mean()
    return ave_acc, ave_prec, ave_rec, ave_f1


def test(model_):
    """
        'B': 0,'M': 1,'E': 2,'S': 3
    """
    text = '这首先是个民族问题，民族的感情问题。'
    hang_ = []
    for wd in text:
        hang_.append(Config().vocab[wd])
    test_tensor = torch.tensor(hang_, dtype=torch.long)
    res = model_(test_tensor)
    res = res.detach().numpy()
    [print(np.argmax(r)) for r in res]
    # print(res)


if __name__ == '__main__':
    # model = torch.load('cut.bin')
    # test(model)
    # sys.exit()  # 测试分词效果

    torch.manual_seed(1)
    print(torch.cuda.is_available())  # 判断是否存在GPU可用，存在返回True
    device = torch.device('cpu')  # 使用CPU进行训练
    # device = torch.device('cuda:0')  # 使用GPU进行训练
    config = Config()
    voc_size = len(config.vocab)

    train_data_list = []
    for lin in config.train_data:
        hang = []
        for word in lin:
            hang.append(config.vocab[word])
        train_data_list.append(torch.tensor(hang, dtype=torch.long))

    target_dict = {'B': 0,
                   'M': 1,
                   'E': 2,
                   'S': 3}
    target_list = []
    for lin in config.target:
        hang = []
        for tag in lin:
            hang.append(target_dict[tag])
        target_list.append(torch.tensor(hang, dtype=torch.long))

    model = Model(config.output_size, voc_size, config.embed_dim)
    model.to(device)  # 将模型使用GPU训练
    losses = []
    acc = []
    rec = []
    prec = []
    f1 = []
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    loss_f = nn.CrossEntropyLoss()
    for i in tqdm(range(config.epoch)):
        for j, k in enumerate(train_data_list):
            optimizer.zero_grad()
            out = model(k.to(device))
            loss = loss_f(out, target_list[j].to(device))
            loss.backward()
            optimizer.step()
            acc_, prec_, rec_, f1_ = model_eval(out, target_list[j])
            acc.append(acc_.item())
            prec.append(prec_.item())
            rec.append(rec_.item())
            f1.append(f1_.item())
            # print('acc: ' + str(acc_.item()) + '\tprec: ' + str(prec_.item()) + '\trec: ' + str(
            #     rec_.item()) + '\tf1: ' + str(f1_.item()))
            losses.append(loss.item())
        torch.save(model, './cut.bin')
        print('acc: ' + str(torch.tensor(acc).mean().item()) + '\tprec: ' + str(torch.tensor(prec).mean().item())
              + '\trec: ' + str(torch.tensor(rec).mean().item()) + '\tf1: ' + str(torch.tensor(f1).mean().item()))
