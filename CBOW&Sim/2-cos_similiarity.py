'''
使用训练的CBOW模型进行相似度计算
'''
import torch
import pickle as pkl
import jieba


def get_word_embed(sentence):
    model = torch.load('data/CBOW.bin')  # 模型实例化
    embed = model.embedding(sentence)  # 将句子送入模型的embedding层，获取embedding后的句子
    wd_tensors = model.layer1(embed)  #
    return wd_tensors


def test_():
    # 准备数据
    sentences1 = '海啸发生在当地时间１７日晚８时许，由在该国北海岸发生的一次地震引发。'
    sentences2 = '这次海啸是由在该国北海岸发生的一次里氏７级海底地震引发的。'
    sentences3 = '韩日元贬值还使东南亚国家的出口严重受挫。'
    sents = [sentences1, sentences2, sentences3]
    word1 = '贬值'
    word2 = '总裁'
    word3 = '总统'
    words_sim = [word1, word2, word3]
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    with open('data/vocab.pkl', 'rb') as f:
        vocab_ = pkl.load(f)
    words_ = []
    for sent in sents:
        words = jieba.lcut(sent)
        temp = []
        for wd in words:
            if wd not in stopwords:
                temp.append(wd)
        words_.append(temp)
    sents_id = []
    for sent in words_:
        sent_id = []
        for wd in sent:
            sent_id.append(vocab_[wd])
        sents_id.append(sent_id)

    words_tensor = torch.tensor([vocab_[wd] for wd in words_sim], dtype=torch.long)
    sents_tensor = [torch.tensor(idx, dtype=torch.long) for idx in sents_id]
    word_embed = get_word_embed(words_tensor)
    sent_embed = []
    for s in sents_tensor:
        sent_embed.append(get_word_embed(s))

    # 计算相似度
    max_length = max(sent_embed[0].view(1, -1).shape[1],
                     sent_embed[1].view(1, -1).shape[1],
                     sent_embed[2].view(1, -1).shape[1], )
    t1 = torch.zeros(1, max_length - sent_embed[0].view(1, -1).shape[1])
    t2 = torch.zeros(1, max_length - sent_embed[1].view(1, -1).shape[1])
    t3 = torch.zeros(1, max_length - sent_embed[2].view(1, -1).shape[1])
    sent1 = torch.cat((sent_embed[0].view(1, -1), t1), dim=1)
    sent2 = torch.cat((sent_embed[1].view(1, -1), t2), dim=1)
    sent3 = torch.cat((sent_embed[2].view(1, -1), t3), dim=1)
    sim_1_2 = torch.cosine_similarity(sent1, sent2)
    sim_2_3 = torch.cosine_similarity(sent2, sent3)
    sim_wd1 = torch.cosine_similarity(word_embed[0].view(1, -1), word_embed[2].view(1, -1))
    sim_wd2 = torch.cosine_similarity(word_embed[1].view(1, -1), word_embed[2].view(1, -1))
    print('句子1和句子2的相似度是：{}'.format(sim_1_2.item()))
    print('句子2和句子3的相似度是：{}'.format(sim_2_3.item()))
    print('词语1和词语2的相似度是：{}'.format(sim_wd1.item()))
    print('词语2和词语3的相似度是：{}'.format(sim_wd2.item()))

    odis = torch.nn.PairwiseDistance(p=2)
    odis_sen1_2 = odis(sent1, sent2)
    odis_sen2_3 = odis(sent2, sent3)
    odis_wd1 = odis(word_embed[0].view(1, -1), word_embed[2].view(1, -1))
    odis_wd2 = odis(word_embed[1].view(1, -1), word_embed[2].view(1, -1))
    print('句子1和句子2的欧式距离是：{}'.format(odis_sen1_2.item()))
    print('句子2和句子3的欧氏距离是：{}'.format(odis_sen2_3.item()))
    print('词语1和词语2的欧氏距离是：{}'.format(odis_wd1.item()))
    print('词语2和词语3的欧氏距离是：{}'.format(odis_wd2.item()))


if __name__ == '__main__':
    test_()
