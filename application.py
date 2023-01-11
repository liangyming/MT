import pickle
import torch
from models import Seq2Seq
import config


def translate(model_path):
    en_lang = pickle.load(open('./save/en_lang.pkl', 'rb'))
    zh_lang = pickle.load(open("./save/zh_lang.pkl", 'rb'))
    model = Seq2Seq(
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden,
        input_vocab_size=en_lang.n_words,
        out_vocab_size=zh_lang.n_words
    ).to(config.device)
    model.load_state_dict(
        torch.load(model_path)
    )
    model.eval()
    while True:
        text = input(">>>: ")
        if text == 'q':
            print('感谢使用!')
            break
        text = text.split()
        text = [en_lang.word2index[word] for word in text]
        input_len = torch.tensor([len(text)])
        input_list = text + [config.PAD_token] * (config.MAX_len - input_len.item())
        input_seq = torch.LongTensor(input_list).unsqueeze(0).to(config.device)
        predict = model.beam_search(input_seq, input_len, config.beam_width)
        result = "".join([zh_lang.index2word[index] for index in predict])
        print("-->: " + result + "\n")