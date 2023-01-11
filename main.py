from utils import Lang
from mydata import get_data
from trainer import train


if __name__ == '__main__':
    en_lang = Lang('en')
    zh_lang = Lang('zh')
    train_data, valid_data, test_data = get_data(en_lang, zh_lang)
    model = train(en_lang, zh_lang, train_data, valid_data)




