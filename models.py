import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import BeamSearch
import config
import random


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=config.PAD_token
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, input, input_len):
        '''
        :param input: [batch_size, seq_len]
        :param input_len: [batch_size]
        :return:
        '''
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        embedded = pack_padded_sequence(embedded, input_len, batch_first=True)
        output, hidden = self.gru(embedded, hidden)
        output, output_len = pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, input):
        batch_size = input.size(0)
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden.to(config.device)


class AttnDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, n_layers=1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=config.PAD_token
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attn_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = Attention(method='concat', hidden_size=hidden_size)

    def forward(self, target, encoder_hidden, encoder_outputs):
        '''
        :param target: [batch_size, max_len + 1]
        :param encoder_hidden: [n_layers=1, batch_size, hidden_size]
        :param encoder_outputs: [batch_size, seq_len, hidden_size]
        :return:
        '''
        batch_size = target.size(0)
        max_decoder_len = config.MAX_len + 1
        decoder_input = torch.LongTensor([[config.SOS_token]] * batch_size).to(config.device)
        # [batch_size, seq_len, vocab_size]保存每个time step结果
        decoder_results = torch.zeros([
            batch_size,
            max_decoder_len,
            self.vocab_size
        ]).to(config.device)
        decoder_hidden = encoder_hidden
        if random.random() > config.teacher_forcing:
            for t in range(max_decoder_len):
                decoder_output_t, decoder_hidden = self.forward_step(
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_outputs=encoder_outputs
                )
                decoder_results[:, t, :] = decoder_output_t
                decoder_input = target[:, t].unsqueeze(-1)
        else:
            for t in range(max_decoder_len):
                decoder_output_t, decoder_hidden = self.forward_step(
                    decoder_input=decoder_input,
                    decoder_hidden=decoder_hidden,
                    encoder_outputs=encoder_outputs
                )
                decoder_results[:, t, :] = decoder_output_t
                decoder_index = torch.argmax(decoder_output_t, dim=-1)
                decoder_input = decoder_index.unsqueeze(-1)
        return decoder_results, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        '''
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [n_layers=1, batch_size, hidden_size]
        :param encoder_outputs: [batch_size, seq_len, hidden_size]
        :return: decoder_output_t->[batch_size, vocab_size]
        '''
        embedded = self.embedding(decoder_input)
        output, decoder_hidden = self.gru(embedded, decoder_hidden)
        attn_weight = self.attention(decoder_hidden, encoder_outputs)
        context_vector = torch.bmm(attn_weight.unsqueeze(1), encoder_outputs)
        concat = torch.cat((output, context_vector), dim=-1)  # [batch_size, 1, hidden_size * 2]
        attn_result = F.tanh(self.attn_fc(concat.squeeze(1)))  # [batch_size, hidden_size]
        decoder_output_t = F.log_softmax(self.fc(attn_result), dim=-1)
        return decoder_output_t, decoder_hidden

    def evaluate(self, encoder_hidden, encoder_outputs):
        '''
        :param encoder_hidden: [n_layers=1, batch_size, hidden_size]
        :param encoder_outputs: [batch_size, seq_len, hidden_size]
        :return:
        '''
        batch_size = encoder_outputs.size(0)
        max_decoder_len = config.MAX_len + 1
        decoder_input = torch.LongTensor([[config.SOS_token]] * batch_size).to(config.device)
        # [batch_size, seq_len, vocab_size]保存每个time step结果
        decoder_results = torch.zeros([
            batch_size,
            max_decoder_len,
            self.vocab_size
        ]).to(config.device)
        decoder_hidden = encoder_hidden
        for t in range(max_decoder_len):
            decoder_output_t, decoder_hidden = self.forward_step(
                decoder_input=decoder_input,
                decoder_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs
            )
            decoder_results[:, t, :] = decoder_output_t
            decoder_index = torch.argmax(decoder_output_t, dim=-1)
            decoder_input = decoder_index.unsqueeze(-1)
        return decoder_results

    def beam_search(self, encoder_hidden, encoder_outputs, beam_width):
        '''
        :param encoder_hidden: [n_layers=1, batch_size=1, hidden_size]
        :param encoder_outputs: [batch_size=1, seq_len, hidden_size]
        :param beam_width: 3
        :return: [word_index]
        '''
        max_decoder_len = config.MAX_len + 1
        decoder_input = torch.LongTensor([[config.SOS_token]]).to(config.device)
        decoder_hidden = encoder_hidden
        prev_beam = BeamSearch(beam_width)
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)
        while True:
            cur_beam = BeamSearch(beam_width)
            for score, flag, seqs, decoder_input, decoder_hidden in prev_beam:
                if flag:
                    cur_beam.add(score, flag, seqs, decoder_input, decoder_hidden)
                else:
                    decoder_output_t, decoder_hidden = self.forward_step(
                        decoder_input=decoder_input,
                        decoder_hidden=decoder_hidden,
                        encoder_outputs=encoder_outputs
                    )
                    value, index = torch.topk(decoder_output_t, k=beam_width)
                    for val, idx in zip(value[0], index[0]):
                        cur_score = score * val.item()
                        cur_seqs = seqs.append(idx)
                        decoder_input = torch.tensor([[idx]], device=config.device)
                        if idx.item() == config.EOS_token:
                            cur_flag = True
                        else:
                            cur_flag = False
                        cur_beam.add(cur_score, cur_flag, cur_seqs, decoder_input, decoder_hidden)
                best_score, best_flag, best_seqs, _, _ = max(cur_beam)
                if best_flag or (len(best_seqs) == max_decoder_len + 1):
                    best_seqs = [i.item() for i in best_seqs]
                    if best_seqs[0] == config.SOS_token:
                        best_seqs = best_seqs[1:]
                    if best_seqs[-1] == config.EOS_token:
                        best_seqs = best_seqs[:-1]
                    return best_seqs
                else:
                    prev_beam = cur_beam

    def init_hidden(self, input):
        batch_size = input.size(0)
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden.to(config.device)


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            # general为对decoder_hidden 进行矩阵变换后，与encoder_outputs相乘
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == 'concat':
            self.Wa = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        '''
        :param decoder_hidden: [n_layers=1, batch_size, hidden_size]
        :param encoder_outputs: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len]
        '''
        if self.method == 'dot':
            # 取出编码器hidden的最后一层
            hidden = decoder_hidden[-1, :, :].unsqueeze(2)  # [batch_size, hidden_size, 1]
            attn_weight = torch.bmm(encoder_outputs, hidden)  # [batch_size, seq_len, 1]
            attn_weight = F.log_softmax(attn_weight.squeeze(-1), dim=-1)
            return attn_weight

        elif self.method == 'general':
            hidden = decoder_hidden[-1, :, :]  # [batch_size, hidden_size]
            hidden = self.Wa(hidden).unsqueeze(2)  # [batch_size, hidden_size, 1]
            attn_weight = torch.bmm(encoder_outputs, hidden)  # [batch_size, seq_len, 1]
            attn_weight = F.log_softmax(attn_weight.squeeze(-1), dim=-1)
            return attn_weight

        elif self.method == 'concat':
            batch_size, encoder_seq_len, _ = encoder_outputs.size()
            hidden = decoder_hidden[-1, :, :]  # [batch_size, hidden_size]
            hidden = hidden.repeat(encoder_seq_len, 1, 1).transpose(1, 0)  # [batch_size, seq_len, hidden_size]
            # concat: [batch_size * seq_len, en_hidden + de_hidden]
            concat = torch.cat((hidden, encoder_outputs), dim=-1).view(batch_size*encoder_seq_len, -1)
            concat = torch.tanh(self.Wa(concat))  # [batch_size * seq_len, hidden_size]
            attn_weight = self.Va(concat)  # [batch_size * seq_len, 1]
            attn_weight = F.log_softmax(
                attn_weight.view([batch_size, encoder_seq_len]),
                dim=-1
            )
            return attn_weight


class Seq2Seq(nn.Module):
    def __init__(self, embedding_dim, hidden_size, input_vocab_size, out_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            vocab_size=input_vocab_size
        ).to(config.device)
        self.decoder = AttnDecoder(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            vocab_size=out_vocab_size
        ).to(config.device)

    def forward(self, input, target, input_len):
        encoder_out, encoder_hidden = self.encoder(input, input_len)
        decoder_out, _ = self.decoder(target, encoder_hidden, encoder_out)
        return decoder_out

    def evaluation(self, input, input_len):
        encoder_out, encoder_hidden = self.encoder(input, input_len)
        decoder_out = self.decoder.evaluate(encoder_hidden, encoder_out)
        return decoder_out

    def beam_search(self, input, input_len, beam_width):
        encoder_out, encoder_hidden = self.encoder(input, input_len)
        best_seq = self.decoder.beam_search(encoder_hidden, encoder_out, beam_width)
        return best_seq

