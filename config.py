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
SOS_token = 0
EOS_token = 1
MAX_len = 17
train_path = 'data/train.txt'
valid_path = 'data/valid.txt'
test_path = 'data/test.txt'
save_path = 'result/'
teacher_forcing = 0.5
lr = 0.01
hidden = 256
epochs = 10

