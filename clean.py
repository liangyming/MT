import config
import re


def check(str):
    my_re = re.compile(r'[A-Za-z]', re.S)
    res = re.findall(my_re, str)
    if len(res):
        return True
    else:
        return False


def clean_data(filename):
    with open(filename + '.en', 'r', encoding='utf-8') as en_data:
        with open(filename + '.zh', 'r', encoding='utf-8') as zh_data:
            with open(filename + '.txt', 'w', encoding='utf-8') as data:
                en = en_data.readlines()
                zh = zh_data.readlines()
                for index, (e, z) in enumerate(zip(en, zh)):
                    z = z.replace(' ', '')
                    if ('（' in z) or ('）' in z) or ('：' in z) or ('·' in z) or ('.' in z) or ('《' in z) or ('》' in z) or (len(z) < 13) or (len(z) > 17):
                        continue
                    if (bool(re.search(r'\d', z)) or check(z)) and len(z) < 16:
                        continue
                    data.write(e.strip('\r\n'))
                    data.write('\t')
                    data.write(z)


if __name__ == '__main__':
    # clean_data('./data/train')
    # clean_data('./data/valid')
    clean_data('./data/test')

