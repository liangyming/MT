import re
import unicodedata
import config


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'   # and c in all_letters
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,])", r" ", s)
    s = re.sub(r"([.!?])", r" \1", s)  # 去除.!?，但是加上\1后又添加了' '和原符号
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {
            'SOS': config.SOS_token,
            'EOS': config.EOS_token,
            'UNK': config.UNK_token,
            'PAD': config.PAD_token
        }
        self.word2count = {}
        self.index2word = {
            config.SOS_token: 'SOS',
            config.EOS_token: 'EOS',
            config.UNK_token: 'UNK',
            config.PAD_token: 'PAD'
        }
        self.n_words = 4

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)
