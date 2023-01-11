import torch
import logging


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(
    filename='training.log',
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)
logger = logging.getLogger('__file__')
'''
0: 'SOS',
1: 'EOS',
2: 'UNK',
3: 'PAD'
'''
PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3
MAX_len = 17
train_path = 'data/train.txt'
valid_path = 'data/valid.txt'
test_path = 'data/test.txt'
save_path = 'result/'
teacher_forcing = 0.5
beam_width = 3
batch_size = 64
lr = 5e-4
embedding_dim = 256
hidden = 256
epochs = 20
clip = 1
warm_up_ratio = 0.1

