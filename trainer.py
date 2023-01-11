from models import Seq2Seq
import torch.nn.functional as F
import config
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


def train(en_lang, zh_lang, train_data, valid_data):
    model = Seq2Seq(
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden,
        input_vocab_size=en_lang.n_words,
        out_vocab_size=zh_lang.n_words
    ).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-8)
    total_step = len(train_data) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=total_step,
        num_warmup_steps=config.warm_up_ratio * total_step
    )
    min_train_loss = float("inf")
    min_valid_loss = float("inf")
    total_loss = 0.0
    for epoch in range(config.epochs):
        model.train()
        config.logger.info("======Begin training: {} epoch======".format(epoch))
        bar = tqdm(train_data, desc="Seq2Seq training: ", total=len(train_data))
        for index, (input, input_len, target, target_len) in enumerate(bar):
            optimizer.zero_grad()
            output = model(input, target, input_len)
            loss = F.nll_loss(
                output.view(-1, zh_lang.n_words),
                target.view(-1),
                ignore_index=config.PAD_token
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            scheduler.step()
            bar.set_description(
                'epoch:{},idx:{}/{},loss:{:.6f}'
                .format(epoch + 1, index, len(train_data), loss.item())
            )
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_data)
        config.logger.info("average train loss: {:.6f}".format(avg_train_loss))

        config.logger.info("======Begin valid======")
        avg_valid_loss = eval(model, valid_data, zh_lang.n_words)
        if avg_train_loss < min_train_loss:
            min_train_loss = avg_train_loss
            torch.save(
                model.state_dict(),
                "./save/" + str(min_train_loss)[:5] + "_"
                + str(epoch) + "train_model.pkl"
            )
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            torch.save(
                model.state_dict(),
                "./save/" + str(min_valid_loss)[:5] + "_"
                + str(epoch) + "valid_model.pkl"
            )
    return model


def eval(model, dataloader, out_vocab_size):
    model.eval()
    bar = tqdm(dataloader, desc="Seq2Seq testing: ", total=len(dataloader))
    total_loss = 0.0
    with torch.no_grad():
        for index, (input, input_len, target, target_len) in enumerate(bar):
            output = model.evaluation(input, input_len)
            loss = F.nll_loss(
                output.view(-1, out_vocab_size),
                target.view(-1),
                ignore_index=config.PAD_token
            )
            bar.set_description(
                'index:{}/{},loss:{:.6f}'
                .format(index, len(dataloader), loss.item())
            )
            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            config.logger.info("average valid loss: {:.6f}".format(avg_loss))
    return avg_loss

