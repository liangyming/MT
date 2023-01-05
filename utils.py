import re
import unicodedata


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
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            0: 'SOS',
            1: 'EOS',
            2: 'UNK',
            3: 'PAD'
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
        for word in sentence.split(' '):
            self.add_word(word)
