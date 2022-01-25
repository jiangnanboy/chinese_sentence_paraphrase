import torch
import torch.nn as nn
from gensim.models import KeyedVectors

import os

def get_token_embedding(vocab_size, hidden_size, token2idx, idx2token, pretrained_embeddings_file, padding_idx):
    '''
    token embedding and init
    :param vocab_size:
    :param hidden_size:
    :param token2idx:
    :param idx2token:
    :param pretrained_embeddings_file:
    :param padding_idx:
    :return:
    '''
    if pretrained_embeddings_file is None or not os.path.exists(pretrained_embeddings_file):
        print('# define embedding.')
        token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    else:
        print("# embedding is init from file {}.".format(pretrained_embeddings_file))
        wvmodel = KeyedVectors.load_word2vec_format(pretrained_embeddings_file)
        embed_size = 300
        weight = torch.zeros(vocab_size, embed_size)
        for i in range(len(wvmodel.index_to_key)):
            try:
                index = token2idx[wvmodel.index2word[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx2token[index]))

        token_embedding = nn.Embedding.from_pretrained(embeddings=weight, padding_idx=padding_idx)

    return token_embedding

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs























