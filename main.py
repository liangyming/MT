from models import *
from torch import optim, nn
import config
import torch
import tqdm
from utils import Lang
from mydata import get_data


def train(en_lang, zh_lang, train_data, valid_data):
    pass


if __name__ == '__main__':
    en_lang = Lang('en')
    zh_lang = Lang('zh')
    train_data, valid_data, test_data = get_data(en_lang, zh_lang)
    train(en_lang, zh_lang, train_data, valid_data)
    '''
    from transformers import BertTokenizer, BertModel

    name = 'bert-base-uncased'
    sentence = "I love you very much!"
    tokenize = BertTokenizer.from_pretrained(name)
    seq = tokenize.encode(sentence, return_tensors='pt')
    bert = BertModel.from_pretrained(name)
    out = bert(seq)
    last_hidden, pools = out
    print(last_hidden)
    print(pools)
    '''



