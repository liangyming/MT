import jieba
import torch
import config
from utils import *


def read_lang(en_lang, zh_lang, path):
    lines = open(path, 'r', encoding='utf-8').readlines()
    pairs = []
    for line in lines:
        line = line.split('\t')
        en_seq = normalizeString(line[0])
        zh_seq = line[1]
        seq_list = jieba.cut(zh_seq, cut_all=False)
        zh_seq = ' '.join(seq_list)
        en_list = en_seq.split(' ')
        zh_list = zh_seq.split(' ')
        if len(en_list) > 17 or len(zh_list) > 17:
            continue
        pairs.append([en_seq, zh_seq])
        en_lang.add_sentence(en_seq)
        zh_lang.add_sentence(zh_seq)
    return pairs


def seq2tensor(en_lang, zh_lang, pairs):
    x_data = []
    y_data = []
    for pair in pairs:
        english = [en_lang.word2index[word] for word in pair[0].split(' ')]
        english.append(config.EOS_token)
        chinese = [zh_lang.word2index[word] for word in pair[1].split(' ')]
        chinese.append(config.EOS_token)
        x_data.append(torch.LongTensor(english).view(-1, 1).to(config.device))
        y_data.append(torch.LongTensor(chinese).view(-1, 1).to(config.device))
    return (x_data, y_data)


def get_data(en_lang, zh_lang):
    train_data = read_lang(en_lang, zh_lang, config.train_path)
    valid_data = read_lang(en_lang, zh_lang, config.valid_path)
    test_data = read_lang(en_lang, zh_lang, config.test_path)
    config.logger.info("english vocab_size: {}".format(en_lang.n_words))
    config.logger.info("chinese vocab_size: {}".format(zh_lang.n_words))
    config.logger.info("train data size: {}".format(len(train_data)))
    config.logger.info("valid data size: {}".format(len(valid_data)))
    config.logger.info("test data size: {}".format(len(test_data)))
    train_data = seq2tensor(en_lang, zh_lang, train_data)
    valid_data = seq2tensor(en_lang, zh_lang, valid_data)
    test_data = seq2tensor(en_lang, zh_lang, test_data)
    return train_data, valid_data, test_data


# if __name__ == '__main__':
#     en_lang = Lang('en')
#     zh_lang = Lang('zh')
#     train_data, valid_data, test_data = get_data(en_lang, zh_lang)
#     print(test_data)
