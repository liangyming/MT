from models import *
from torch import optim, nn
import config
import torch
import tqdm
from utils import Lang
from mydata import get_data


def train(en_lang, zh_lang, train_data, valid_data):
    encoder = Encoder(en_lang.n_words, config.hidden).to(config.device)
    decoder = Attention(hidden_size=config.hidden,
                        output_size=zh_lang.n_words,
                        max_length=config.MAX_len + 1,
                        dropout_p=0.1)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=config.lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=config.lr)
    scheduler_encoder = optim.lr_scheduler.StepLR(encoder_optimizer,
                                                  step_size=1,
                                                  gamma=0.95)
    scheduler_decoder = optim.lr_scheduler.StepLR(decoder_optimizer,
                                                  step_size=1,
                                                  gamma=0.95)
    criterion = nn.NLLLoss()
    for epoch in tqdm.tqdm(range(config.epochs)):
        config.logger.info("train {} epoch data".format(epoch))
        for index, (x, y) in enumerate(zip(train_data)):
            loss = loss_func(
                input=x,
                output=y,
                encoder=encoder,
                decoder=decoder,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                criterion=criterion
            )
            if index % 200 == 0:
                config.logger.info("{} epoch {} times loss: {}".format(epoch, index, loss))
        torch.save(encoder.state_dict(), config.save_path + str(epoch) + 'encode.pth')
        torch.save(decoder.state_dict(), config.save_path + str(epoch) + 'decode.pth')
        scheduler_encoder.step()
        scheduler_decoder.step()


if __name__ == '__main__':
    en_lang = Lang('en')
    zh_lang = Lang('zh')
    train_data, valid_data, test_data = get_data(en_lang, zh_lang)
    train(en_lang, zh_lang, train_data, valid_data)


