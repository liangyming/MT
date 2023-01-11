import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
import pickle
import config


class MTDataset(Dataset):
    def __init__(self, en_lang, zh_lang, pairs):
        super(MTDataset, self).__init__()
        self.x_data = []
        self.y_data = []
        for pair in pairs:
            english = [en_lang.word2index[word] for word in pair[0]]
            # english.append(config.EOS_token)
            chinese = [zh_lang.word2index[word] for word in pair[1]]
            chinese.append(config.EOS_token)
            self.x_data.append(english)
            self.y_data.append(chinese)

    def __getitem__(self, index):
        return self.x_data[index], len(self.x_data[index]),\
               self.y_data[index], len(self.y_data[index])

    def __len__(self):
        return len(self.x_data)


def collate_fn(batch):
    def merge(datas, lengths, max_len):
        padded_seqs = torch.zeros(len(datas), max_len).long()
        for i, seq in enumerate(datas):
            seq = torch.tensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs.to(config.device)

    batch.sort(key=lambda x: x[1], reverse=True)
    input_seq, input_len, out_seq, out_len = zip(*batch)
    input_seq = merge(input_seq, input_len, config.MAX_len)
    # 输出句子末尾加了EOS, max_len加一
    out_seq = merge(out_seq, out_len, config.MAX_len + 1)
    input_len = torch.tensor(input_len)
    out_len = torch.tensor(out_len)
    return input_seq, input_len, out_seq, out_len


def read_lang(en_lang, zh_lang, path):
    lines = open(path, 'r', encoding='utf-8').readlines()
    pairs = []
    for line in lines:
        line = line.strip().split('\t')
        en_seq = normalizeString(line[0])
        zh_seq = line[1]
        zh_seq = jieba.lcut(zh_seq, cut_all=False)
        en_seq = en_seq.split()
        if len(en_seq) > 17 or len(zh_seq) > 17:
            continue
        pairs.append([en_seq, zh_seq])
        en_lang.add_sentence(en_seq)
        zh_lang.add_sentence(zh_seq)
    return pairs


def get_data(en_lang, zh_lang):
    train_data = read_lang(en_lang, zh_lang, config.train_path)
    # max_len = max([len(pair[0]) for pair in train_data])
    valid_data = read_lang(en_lang, zh_lang, config.valid_path)
    test_data = read_lang(en_lang, zh_lang, config.test_path)
    config.logger.info("english vocab_size: {}".format(en_lang.n_words))
    config.logger.info("chinese vocab_size: {}".format(zh_lang.n_words))
    config.logger.info("train data size: {}".format(len(train_data)))
    config.logger.info("valid data size: {}".format(len(valid_data)))
    config.logger.info("test data size: {}".format(len(test_data)))
    # 保存词表
    pickle.dump(en_lang, open("./save/en_lang.pkl", 'wb'))
    pickle.dump(zh_lang, open("./save/zh_lang.pkl", 'wb'))

    train_set = MTDataset(en_lang, zh_lang, train_data)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_set = MTDataset(en_lang, zh_lang, valid_data)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_set = MTDataset(en_lang, zh_lang, test_data)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_loader, valid_loader, test_loader


# if __name__ == '__main__':
#     from models import Encoder
#     en_lang = Lang('en')
#     zh_lang = Lang('zh')
#     train_loader, valid_loader, test_loader = get_data(en_lang, zh_lang)
#     encoder = Encoder(config.embedding_dim, config.hidden, en_lang.n_words).to(config.device)
#     for i, (input_seq, input_len, out_seq, out_len) in enumerate(train_loader):
#         output, hidden = encoder(input_seq, input_len)
#         print(input_seq)
#         print(input_len)
#         print(out_seq)
#         print(out_len)
