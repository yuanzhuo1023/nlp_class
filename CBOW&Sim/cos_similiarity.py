import torch
import pickle as pkl
import jieba

def get_word_embed(sentence):
    model = torch.load('data/CBOW.bin')
    embed = model.embedding(sentence)
    wd_tensors = model.layer1(embed)
    return wd_tensors

def test_embed():
    # 数据处理
    sentences1 = '海啸发生在当地时间１７日晚８时许。'
    sentences2 = '这次海啸是由在该国北海岸发生的一次里氏７级海底地震引发的。'
    sentences3 = '韩日元贬值还使东南亚国家的出口严重受挫。'
    sens = [sentences1, sentences2, sentences3]
    word1 = '贬值'
    word2 = '总裁'
    word3 = '总统'
    words_sim = [word1, word2, word3]
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    with open('data/vocab.pkl', 'rb') as f:
        _vocab = pkl.load(f)
    words_ = []
    for sen in sens:
        words = jieba.lcut(sen)
        temp = []
        for wd in words:
            if wd not in stopwords:
                temp.append(wd)
        words_.append(temp)

    sens_id = []
    for sen in words_:
        sen_id = []
        for wd in sen:
            sen_id.append(_vocab[wd])
        sens_id.append(sen_id)

    # 将数据转为张量形式
    words_tensor = torch.tensor([_vocab[wd] for wd in words_sim], dtype=torch.long)
    sens_tensor = [torch.tensor(idx, dtype=torch.long) for idx in sens_id]
    word_embed = get_word_embed(words_tensor)
    sen_embed = []
    for s in sens_tensor:
        len_s = len(s)
        #一个句子有len_s个词，即有len_s*32的张量
        s_1 = get_word_embed(s)
        #将句子变为1*32维的张量，每一列取平均值，得到一个句子的32维张量表示
        s_2 = s_1.sum(0)/len_s
        sen_embed.append(s_2)

    sim_sen1 = torch.cosine_similarity(sen_embed[0].view(1, -1), sen_embed[1].view(1, -1))
    sim_sen2 = torch.cosine_similarity(sen_embed[1].view(1, -1), sen_embed[2].view(1, -1))
    sim_wd1 = torch.cosine_similarity(word_embed[0].view(1, -1), word_embed[2].view(1, -1))
    sim_wd2 = torch.cosine_similarity(word_embed[1].view(1, -1), word_embed[2].view(1, -1))
    print('句子1和句子2的相似度是：{}'.format(sim_sen1.item()))
    print('句子2和句子3的相似度是：{}'.format(sim_sen2.item()))
    print('词语1和词语2的相似度是：{}'.format(sim_wd1.item()))
    print('词语2和词语3的相似度是：{}'.format(sim_wd2.item()))

    odis = torch.nn.PairwiseDistance(p=2)
    odis_sen1 = odis(sen_embed[0].view(1, -1), sen_embed[1].view(1, -1))
    odis_sen2 = odis(sen_embed[0].view(1, -1), sen_embed[2].view(1, -1))
    odis_wd1 = odis(word_embed[0].view(1, -1), word_embed[2].view(1, -1))
    odis_wd2 = odis(word_embed[1].view(1, -1), word_embed[2].view(1, -1))
    print('句子1和句子2的欧式距离是：{}'.format(odis_sen1.item()))
    print('句子2和句子3的欧氏距离是：{}'.format(odis_sen2.item()))
    print('词语1和词语2的欧氏距离是：{}'.format(odis_wd1.item()))
    print('词语2和词语3的欧氏距离是：{}'.format(odis_wd2.item()))

if __name__ == '__main__':
    test_embed()
