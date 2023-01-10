from models import Seq2Seq
from torch import optim
import torch.nn.functional as F
import config
import torch
import tqdm
from utils import Lang
from mydata import get_data


def train(en_lang, zh_lang, train_data, valid_data):
    model = Seq2Seq(
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden,
        input_vocab_size=en_lang.n_words,
        out_vocab_size=zh_lang.n_words
    ).to(config.device)
    optimizer = optim.Adam(model.parameters())
    model.train()
    bar = tqdm.tqdm(train_data, desc="Seq2Seq training***", total=len(train_data))
    for index, (input, input_len, target, target_len) in enumerate(bar):
        optimizer.zero_grad()
        output = model(input, target, input_len)
        loss = F.nll_loss(
            target.view(-1, zh_lang.n_words),
            output.view(-1),
            ignore_index=config.PAD_token
        )
        loss.backward()
        optimizer.step()


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



