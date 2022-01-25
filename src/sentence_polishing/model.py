import torch
import torch.nn as nn

import math

from .modules import get_token_embedding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        assert hid_dim % heads == 0
        self.hid_dim = hid_dim
        self.heads = heads
        self.head_dim = hid_dim // heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)

    def forward(self, query, key, value, mask = None):
        '''
        :param query: [batch_size, query_len, hid_dim]
        :param key: [batch_size, key_len, hid_dim]
        :param value: [batch_size, value_len, hid_dim]
        :param mask:
        :return:
        '''
        batch_size = query.shape[0]
        # [batch_size, query_len, hid_dim]
        Q = self.fc_q(query)
        # [batch_size, key_len, hid_dim]
        K = self.fc_k(key)
        # [batch_size, value_len, hid_dim]
        V = self.fc_v(value)

        # [batch_size, heads, query_len, hid_dim]
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, heads, key_len, hid_dim]
        K = K.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # [batch_size, heads, value_len, hid_dim]
        V = V.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # [batch_size, heads, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # [batch_size, heads, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)
        # [batch_size, heads, query_len, head_dim]
        x = torch.matmul(self.dropout(attention), V)
        # [batch_size, query_len, heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # [batch_size, query_len, hid_dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # [batch_size, query_len, hid_dim]
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x: [batch_size, seq_len, hid_dim]
        :return:
        '''
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class PositionEncoding(nn.Module):
    def __init__(self, hid_dim, max_len=100, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hid_dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-math.log(10000.0) / hid_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x_input):
        '''
        :param x_input: [seq_len, batch_size, hid_dim]
        :return:
        '''
        self.pe = self.pe.to(DEVICE)
        x_input = x_input + self.pe[:x_input.size(0), :]
        return self.dropout(x_input)

class SynonymPositionEncoding(nn.Module):
    def __init__(self, hid_dim, max_len=100):
        super(SynonymPositionEncoding, self).__init__()
        pe = torch.zeros(max_len, hid_dim)
        positioin = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2).float() * (-math.log(10000.0) / hid_dim))
        pe[:, 0::2] = torch.sin(positioin * div_term)
        pe[:, 1::2] = torch.cos(positioin * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, synonym_input):
        '''
        :param synonym_input: [seq_len, batch_size]
        :return:
        '''
        self.pe = self.pe.to(DEVICE)
        return self.pe[:synonym_input.size(0), :]

class Encoder(nn.Module):
    def __init__(self, tok_embedding, hid_dim, layers, heads, pf_dim, dropout, padding_idx, max_length=100):
        super(Encoder, self).__init__()
        self.tok_embedding = tok_embedding
        # self.pos_embedding = nn.Embedding(max_length, hid_dim, padding_idx=padding_idx)
        self.pos_embedding = PositionEncoding(hid_dim, max_length)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(layers)])
        # self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

    def forward(self, src, src_mask):
        '''
        :param src: [batch_size, src_len]
        :param src_mask: [batch_size, 1, 1, src_len]
        :return:
        '''
        '''
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # [batc_size, src_len]
        src = self.dropout((self.tok_embedding(src) * self.scale)) + self.pos_embedding(pos)
        '''

        # [batch_size, src_len, hid_dim]
        src_token_embedding = self.tok_embedding(src) * self.scale
        # [batch_size, src_len, hid_dim]
        src = self.pos_embedding(src_token_embedding.transpose(0, 1)).transpose(0, 1)

        # [batch_size, src_len, hid_dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, heads, pf_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        '''
        :param src: [batch_size, src_len]
        :param src_mask: [batch_size, 1, 1, src_len]
        :return:
        '''
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm, [batch_size, src_len, hid_dim]
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual connection and layer norm, [batch_size, src_len, hid_dim]
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Decoder(nn.Module):
    def __init__(self, tok_embedding, output_dim, hid_dim, layers, heads, pf_dim, dropout, padding_idx, max_length=100):
        super(Decoder, self).__init__()
        self.tok_embedding = tok_embedding
        self.hid_dim = hid_dim
        # self.pos_embedding = nn.Embedding(max_length, hid_dim, padding_idx=padding_idx)
        self.pos_embedding = PositionEncoding(hid_dim, max_length)
        self.synonym_pos_embedding = SynonymPositionEncoding(hid_dim, max_length)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  heads,
                                                  pf_dim,
                                                  dropout)
                                     for _ in range(layers)])
        # self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

        # original_word_parameter_w
        self.W_a_o = nn.Parameter(torch.Tensor(2 * self.hid_dim))
        # original_word_parameter_v
        self.V_a_o = nn.Parameter(torch.Tensor(2 * self.hid_dim))
        # init
        nn.init.normal_(self.W_a_o, std=0.01)
        nn.init.normal_(self.V_a_o, std=0.01)

        # paraphrased_word_parameter_w
        self.W_a_p = nn.Parameter(torch.Tensor(2 * self.hid_dim))
        # paraphrased_word_parameter_v
        self.V_a_p = nn.Parameter(torch.Tensor(2 * self.hid_dim))
        # init
        nn.init.normal_(self.W_a_p, std=0.01)
        nn.init.normal_(self.V_a_p, std=0.01)

        self.f_h = nn.Linear(3 * self.hid_dim, hid_dim, bias=False)

    def forward(self, decoder_input, enc_src, trg_mask, src_mask, sent_synonym_dict):
        '''
        :param decoder_input: [batch_size, trg_len]
        :param enc_src: [batch_size, src_len, hid_dim]
        :param trg_mask: [batch_size, 1, trg_len, trg_len]
        :param src_mask: [batch_size, 1, 1, src_len]
        :param sent_synonym_dict: [batch_size, synonym_words_len, 2]
        :return:
        '''

        # decoder_input info
        batch_size = decoder_input.shape[0]
        trg_len = decoder_input.shape[1]

        # synonym info
        sent_synonym_lens = sent_synonym_dict.shape[1]
        # sent_synonym_o from synonym dict, sent_synonym_p from the position of synonym in encoder input
        sent_synonym_o, sent_synonym_p = sent_synonym_dict[:, :, 0], sent_synonym_dict[:, :, 1]

        '''
        # [batch_size, trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)
        # [batch_size, trg_len, hid_dim]
        trg = self.dropout((self.tok_embedding(decoder_input) * self.scale) + self.pos_embedding(pos))
        '''

        # [batch_size, trg_len, hid_dim]
        trg_token_embedding = self.tok_embedding(decoder_input) * self.scale
        # [batch_size, trg_len, hid_dim]
        trg = self.pos_embedding(trg_token_embedding.transpose(0, 1)).transpose(0, 1)

        # synonym embedding, [batch_size, sent_synonym_lens, hid_dim]
        sent_synonym_o_embedding = self.tok_embedding(sent_synonym_o)
        # synonym position embedding
        sent_synonym_p_embedding = self.synonym_pos_embedding(sent_synonym_p.transpose(0, 1)).transpose(0, 1)

        # 1. get decoder output
        for layer in self.layers:
            # trg : [batch_size, trg_len, hid_dim], attention: [batch_size, heads, trg_len, src_len]
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # add synonym dictionary attention

        # 2.计算基本decoder的输出和同义词embedding的注意力

        # [batch_size, trg_len, sent_synonym_lens, hid_dim]
        h = torch.ones(batch_size, trg_len, sent_synonym_lens, self.hid_dim).to(DEVICE)
        h = h * trg.unsqueeze(2)

        o_embedding = torch.ones(batch_size, trg_len, sent_synonym_lens, self.hid_dim).to(DEVICE)
        o_embedding = o_embedding * sent_synonym_o_embedding.unsqueeze(1)

        # [batch_size, trg_len, sent_synonym_lens, 2*hid_dim]
        h_o_concat = torch.cat((h, o_embedding), dim=-1)

        # [batch_size, trg_len, sent_synonym_lens, 2*hid_dim]
        score_tem_o = torch.tanh(self.W_a_o * h_o_concat)

        # [batch_size, trg_len, sent_synonym_lens]
        score_o = torch.sum(self.V_a_o * score_tem_o, dim=-1)

        # [batch_size, trg_len, sent_synonym_lens]
        a = torch.softmax(score_o, dim=-1)

        # [batch_size, trg_len, sent_synonym_lens] * [batch_size, sent_synonym_lens, hid_dim]
        c_o = torch.matmul(a, sent_synonym_o_embedding) # [batch_size, trg_len, hid_dim]

        '''
        h = tf.fill([batch_size, trg_len, sent_synonym_lens, self.hid_dim], 1.0) * tf.expand_dims(trg, axis=2)

        o_embeding = tf.fill([batch_size, trg_len, sent_synonym_lens, self.hid_dim], 1.0) * tf.expand_dims(
            sent_synonym_o_embedding, axis=1)

        W_a_o = tf.get_variable("original_word_parameter_w", [2 * self.hid_dim],
                                initializer=tf.initializers.random_normal(
                                    stddev=0.01, seed=None))
        V_a_o = tf.get_variable("original_word_parameter_v", [2 * self.hid_dim],
                                initializer=tf.initializers.random_normal(
                                    stddev=0.01, seed=None))

        h_o_concat = tf.concat([h, o_embeding], -1)  # N, T2, W2, 2*d_model
        score_tem_o = tf.tanh(W_a_o * h_o_concat)  # N, T2, W2, 2*d_model
        score_o = tf.reduce_sum(V_a_o * score_tem_o, axis=-1)  # N, T2, W2
        a = tf.nn.softmax(score_o)  # N, T2, W2
        c_o = tf.matmul(a, sent_synonym_o_embedding)  # (N, T2, W2) * (N, W2, d_model) --> N, T2, d_model
        '''

        # 3.计算基本decoder的输出和同义词对应的句子中词的位置id的embedding的注意力
        p_embedding = torch.ones(batch_size, trg_len, sent_synonym_lens, self.hid_dim).to(DEVICE)
        p_embedding = p_embedding * sent_synonym_p_embedding.unsqueeze(1)

        h_p_concat = torch.cat((h, p_embedding), dim=-1) # [batch_size, trg_len, sent_synonym_lens, 2*hid_dim]
        score_tem_p = torch.tanh(self.W_a_p * h_p_concat) # [batch_size, trg_len, sent_synonym_lens, 2*hid_dim]
        score_p = torch.sum(self.V_a_p * score_tem_p, dim=-1) # [batch_size, trg_len, sent_synonym_lens]

        a = torch.softmax(score_p, dim=-1) # [batch_size, trg_len, sent_synonym_lens]
        # [batch_size, trg_len, sent_synonym_lens] * [batch_size, sent_synonym_lens, hid_dim]
        c_p = torch.matmul(a, sent_synonym_p_embedding) # [batch_size, trg_len, hid_dim]

        '''
        p_embeding = tf.fill([batch_size, trg_len, sent_synonym_lens, self.hid_dim], 1.0) * tf.expand_dims(
            sent_synonym_p_embedding, axis=1)
        W_a_p = tf.get_variable("paraphrased_word_parameter_w", [2 * self.hid_dim],
                                initializer=tf.initializers.random_normal(
                                    stddev=0.01, seed=None))
        V_a_p = tf.get_variable("paraphrased_word_parameter_v", [2 * self.hid_dim],
                                initializer=tf.initializers.random_normal(
                                    stddev=0.01, seed=None))
        h_p_concat = tf.concat([h, p_embeding], -1)  # N, T2, W2, 2*d_model
        score_tem_p = tf.tanh(W_a_p * h_p_concat)  # N, T2, W2, 2*d_model
        
        score_p = tf.reduce_sum(V_a_p * score_tem_p, axis=-1)  # N, T2, W2
        a = tf.nn.softmax(score_p)  # N, T2, W2
        c_p = tf.matmul(a, sent_synonym_p_embedding)  # (N, T2, W2) * (N, W2, d_model) --> N, T2, d_model
        '''


        # 4.连接同义词注意力信息和对应句子中词位置的注意力信息

        c_t = torch.cat((c_o, c_p), dim=-1) # [batch_size, trg_len, 2*hid_dim]

        '''
        c_t = tf.concat([c_o, c_p], axis=-1)  # N, T2, d_model --> N, T2, 2*d_model
        '''

        # 5.结合基本decoder的输出信息和同义词与位置的注意力信息，接一个前馈网络
        out_dec = torch.tanh(self.f_h(torch.cat((trg, c_t), dim=-1))) # [batch_size, trg_len, hid_dim]

        '''
        out_dec = tf.layers.dense(tf.concat([trg, c_t], axis=-1), self.hid_dim, activation=tf.tanh,
                                  use_bias=False, kernel_initializer=tf.initializers.random_normal(
                stddev=0.01, seed=None))
        '''

        # 6.Final linear projection (embedding weights are shared)
        weights = torch.transpose(self.tok_embedding.weight, dim0=0, dim1=1) # [hid_dim, vocab_size]
        logits = torch.einsum('ntd, dk -> ntk', out_dec, weights) # [batch_size, trg_len, vocab_size]
        y_hat = torch.argmax(logits, dim=-1) # [batch_size, trg_len]

        '''
        weights = tf.transpose(self.embeddings)  # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', out_dec, weights)  # (N, T2, vocab_size) -> [batch_size, seq_len, vocab_size]
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))  # [batch_size, seq_len]
        '''

        return logits, y_hat

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, heads, pf_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        '''
        :param trg: [batch_size, trg_len, hid_dim]
        :param enc_src: [batch_size, src_len, hid_dim]
        :param trg_mask: [batch_size, 1, trg_len, trg_len]
        :param src_mask: [batch_size, 1, 1, src_len]
        :return:
        '''
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm, [batch_size, trg_len, hid_dim]
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm, [batch_size, trg_len, hid_dim]
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # [batch_size, trg_len, hid_dim], [batch_size, heads, trg_len, src_len]
        return trg, attention

class Transformer(nn.Module):
    def __init__(self, token2idx, idx2token, args, padding_idx):
        super(Transformer, self).__init__()
        self.args = args
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.padding_idx = padding_idx
        self.args.vocab_size = len(self.token2idx)

        self.token_embedding = get_token_embedding(self.args.vocab_size, self.args.d_model, self.token2idx, self.idx2token, self.args.embedding_file, padding_idx=padding_idx)

        self.encoder = Encoder(self.token_embedding, self.args.d_model, self.args.num_blocks, self.args.num_heads, self.args.d_ff, self.args.dropout, padding_idx, self.args.enc_maxlength)
        self.decoder = Decoder(self.token_embedding, self.args.vocab_size, self.args.d_model, self.args.num_blocks, self.args.num_heads, self.args.d_ff, self.args.dropout, padding_idx, self.args.dec_maxlength)

        self.fc_synonym_out = nn.Linear(self.args.d_model, 2, bias=False)

    def mask_src_mask(self, src):
        '''
        :param src: [batch_size, src_len]
        :return:
        '''
        # [batch_size, 1, 1, src_len]
        src_mask = (src != self.padding_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def mask_trg_mask(self, trg):
        '''
        :param trg: [batch_size, trg_len]
        :return:
        '''
        # [batch_size, 1, 1, trg_len]
        trg_pad_mask = (trg != self.padding_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        # [trg_len, trg_len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=DEVICE)).bool()
        # [batch_size, 1, trg_len, trg_len]　
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def synonym_labeling(self, encode_output):
        '''
        :param encode_output: [batch_size, src_len, hid_dim]
        :return:
        '''
        # [batch_size, src_len, 2]
        synonym_logits = torch.tanh(self.fc_synonym_out(encode_output))
        synonym_hat = torch.argmax(synonym_logits, dim=-1)  # [batch_size, src_len]

        return synonym_logits, synonym_hat

    def encode(self, encode_input):
        '''
        :param encode_input: [batch_size, src_len]
        :return:
        '''
        # [batch_size, 1, 1, src_len]
        src_mask = self.mask_src_mask(encode_input)
        # [batch_size, src_len, hid_dim]
        encode_output = self.encoder(encode_input, src_mask)

        return encode_output, src_mask

    def decode(self, decode_input, src_sent_synonym_dict, encode_output, src_mask):
        '''
        :param decode_input:
        :param src_sent_synonym_dict: [batch_size, synonym_words_len, 2]    [[2,3], [4,6], ..., [7,3]]
                是词典中同义词对应句子中同义词，会有多个词典中同义词对应句子中一个同义词
                其形式如：[[2,3], [4,6], ..., [7,3]]
        :param encode_output:
        :param src_mask:
        :return:
        '''
        # [batch_size, 1, trg_len, trg_len]
        trg_mask = self.mask_trg_mask(decode_input)

        # [batch_size, trg_len, vocab_size], [batch_size, trg_len]
        logits, y_hat = self.decoder(decode_input, encode_output, trg_mask, src_mask, src_sent_synonym_dict)

        return logits, y_hat

    def forward(self, encode_input, decode_input, src_sent_synonym_dict):
        '''
        :param encode_input: [batch_size, src_len]
        :param decode_input: [batch_size, trg_len]
        :param src_sent_synonym_dict: [batch_size, synonym_words_len, 2]    [[2,3], [4,6], ..., [7,3]]
        :return:
        '''
        # 1.encode
        # [batch_size, 1, 1, src_len]
        src_mask = self.mask_src_mask(encode_input)
        # [batch_size, src_len, hid_dim]
        encode_output = self.encoder(encode_input, src_mask)

        # 2.synonym
        # [batch_size, src_len, 2], [batch_size, src_len]
        synonym_logits, synonym_hat = self.synonym_labeling(encode_output)

        # 3.decode
        # [batch_size, 1, trg_len, trg_len]
        trg_mask = self.mask_trg_mask(decode_input)
        # [batch_size, trg_len, vocab_size], [batch_size, trg_len]
        decode_logits, decode_y_hat = self.decoder(decode_input, encode_output, trg_mask, src_mask, src_sent_synonym_dict)

        return decode_logits, decode_y_hat, synonym_logits, synonym_hat





