import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import random


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


class Attention(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=config.MAX_len):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input).view(1, -1)
        embedded = self.dropout(embedded)
        attn_weight = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)),
            dim=1
        )
        attn_applied = torch.bmm(
            attn_weight.unsqueeze(0),
            encoder_output.unsqueeze(0)
        )
        output = torch.cat([embedded[0], attn_applied[0]], dim=1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weight

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


def loss_func(input, output, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input.size(0)
    output_length = output.size(0)
    encoder_outputs = torch.zeros(config.MAX_len + 1, encoder.hidden_size, device=config.device)

    for ei in range(input_length):
         encoder_output, encoder_hidden = encoder(input[ei], encoder_hidden)
         encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[config.SOS_token]], device=config.device)

    loss = 0.0
    if random.random() < config.teacher_forcing:
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output[di])
            decoder_input = output[di]
    else:
        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output[di])
            topV, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, output[di])
            if decoder_input.item() == config.EOS_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length

