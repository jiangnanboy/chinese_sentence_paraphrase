import torch
import torch.nn.functional as F

from torchtext.data.metrics import bleu_score

import tqdm
import time
import math

from .modules import epoch_time, count_parameters, initialize_weights
from .hparams import Hparams
from .model import Transformer
from .data_load import load_vocab
from .dataset import GetDataset, get_dataloader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight
        self.ignore_index = ignore_index

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight, ignore_index=self.ignore_index)
        return self.linear_combination(loss / n, nll)

def train_synonym_label(model, iterator, optimizer, synonym_criterion, clip):
    '''
    :param model:
    :param iterator:
    :param optimizer:
    :param synonym_criterion:
    :param clip:
    :return:
    '''
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        encode_input = batch['encoder_input']
        synonym_label = batch['synonym_label']

        encode_input = encode_input.to(DEVICE)
        synonym_label = synonym_label.to(DEVICE)

        optimizer.zero_grad()
        encode_output, _ = model.encode(encode_input)
        # [batch_size, src_len, 2], [batch_size, src_len]
        synonym_logits, synonym_hat = model.synonym_labeling(encode_output)

        synonym_output_dim = synonym_logits.shape[-1]
        # [batch_size * (trg_len - 1), synonym_output_dim]
        synonym_output = synonym_logits.contiguous().view(-1, synonym_output_dim)
        # [batch_size * (trg_len - 1)]
        synonym_label = synonym_label.contiguous().view(-1)

        synonym_loss = synonym_criterion(synonym_output, synonym_label)

        synonym_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += synonym_loss.item()
    return epoch_loss / len(iterator)

def train(model, iterator, optimizer, synonym_criterion, criterion, clip, l_alpha):
    '''
    :param model:
    :param iterator:
    :param optimizer:
    :param synonym_criterion:
    :param criterion:
    :param clip:
    :param l_alpha:
    :return:
    '''
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):

        encode_input = batch['encoder_input']
        decode_input = batch['decoder_input']  # except </s>
        decode_trg = batch['target']  # except <s>
        synonym_label = batch['synonym_label']
        src_sent_synonym_dict = batch['synonym_dict']

        encode_input = encode_input.to(DEVICE)
        decode_input = decode_input.to(DEVICE)
        decode_trg = decode_trg.to(DEVICE)
        synonym_label = synonym_label.to(DEVICE)
        src_sent_synonym_dict = src_sent_synonym_dict.to(DEVICE)

        optimizer.zero_grad()
        # decode_logits, decode_y_hat: [batch_size, trg_len, vocab_size], [batch_size, trg_len]
        # synonym_logits, synonym_hat: [batch_size, src_len, 2], [batch_size, src_len]
        decode_logits, decode_y_hat, synonym_logits, synonym_hat = model(encode_input, decode_input, src_sent_synonym_dict)

        decode_output_dim = decode_logits.shape[-1]
        # [batch_size * (trg_len - 1), decode_output_dim]
        decode_output = decode_logits.contiguous().view(-1, decode_output_dim)
        # [batch_size * (trg_len - 1)]
        decode_trg = decode_trg.contiguous().view(-1)

        synonym_output_dim = synonym_logits.shape[-1]
        # [batch_size * (trg_len - 1), synonym_output_dim]
        synonym_output = synonym_logits.contiguous().view(-1, synonym_output_dim)
        # [batch_size * (trg_len - 1)]
        synonym_label = synonym_label.contiguous().view(-1)

        # multi task loss，同义词分类的loss；生成的loss
        loss = l_alpha * criterion(decode_output, decode_trg) + (1.0 - l_alpha) * synonym_criterion(synonym_output, synonym_label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluation(model, iterator, criterion):
    '''
    :param model:
    :param iterator:
    :param criterion:
    :return:
    '''
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            encode_input = batch['encoder_input']
            decode_input = batch['decoder_input']  # except </s>
            decode_trg = batch['target']  # except <s>
            src_sent_synonym_dict = batch['synonym_dict']

            encode_input = encode_input.to(DEVICE)
            decode_input = decode_input.to(DEVICE)
            decode_trg = decode_trg.to(DEVICE)
            src_sent_synonym_dict = src_sent_synonym_dict.to(DEVICE)

            encode_output, src_mask = model.encode(encode_input)
            # [batch_size, trg_len - 1, vocab_size], [batch_size, trg_len - 1]
            decode_logits, decode_y_hat = model.decode(decode_input, src_sent_synonym_dict, encode_output, src_mask)
            output_dim = decode_logits.shape[-1]
            # [batch_size * (trg_len - 1), output_dim]
            output = decode_logits.contiguous().view(-1, output_dim)
            # [batch_size * (trg_len - 1)]
            decode_trg = decode_trg.contiguous().view(-1)

            eval_loss = criterion(output, decode_trg)
            epoch_loss += eval_loss.item()
    return epoch_loss / len(iterator)

def inference(model, sentence_tokens, src_sent_synonym_dict, max_len=100):
    '''
    inference : greedy search
    :param model:
    :param sentence_tokens: list
    :param src_sent_synonym_dict:
    :param max_len:
    :return:
    '''
    model.eval()

    sentence_tokens += ['</s>']
    sent_indexes = [model.token2idx[token] for token in sentence_tokens]
    sent_tensor = torch.LongTensor(sent_indexes).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encode_output, src_mask = model.encode(sent_tensor)

    trg_indexes = [model.token2idx['<s>']]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            decode_logits, decode_y_hat = model.decode(trg_tensor, src_sent_synonym_dict, encode_output, src_mask)

        pre_token = decode_y_hat[:, -1].item()
        trg_indexes.append(pre_token)

        if pre_token == model.token2idx['</s>']:
            break
    trg_tensors = [model.idx2token[index] for index in trg_indexes]

    return trg_tensors[1:]

def beam_search():
    pass

def train_model(train_iterator, valid_iterator, token2idx, idx2token, args, padding_idx):
    '''
    :param train_iterator:
    :param valid_iterator:
    :param token2idx:
    :param idx2token:
    :param args:
    :param padding_idx:
    :return:
    '''
    model = Transformer(token2idx, idx2token, args, padding_idx).to(DEVICE)
    print(f'the model has {count_parameters(model):,} trainable parameters')

    # init model weights
    # model.apply(initialize_weights())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    synonym_criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    criterion = LabelSmoothingLoss(smoothing=args.smoothing, ignore_index=padding_idx)

    best_valid_loss = float('inf')

    for epoch in range(args.num_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, synonym_criterion, criterion, args.clip, args.l_alpha)
        valid_loss = evaluation(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_model)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def load_model(token2idx, idx2token, args, padding_idx):
    '''
    :param token2idx:
    :param idx2token:
    :param args:
    :param padding_idx:
    :return:
    '''
    model = Transformer(token2idx, idx2token, args, padding_idx).to(DEVICE)
    model.load_state_dict(torch.load(args.save_model))

    return model

def calculate_bleu(model, src_sents, trg_sents, src_sent_synonym_dict, max_len=100):
    '''
    :param model:
    :param src_sents:
    :param trg_sents:
    :param src_sent_synonym_dict:
    :param max_len:
    :return:
    '''
    trgs = []
    pred_trgs = []

    for src_sent_tokens, trg_sent_tokens in zip(src_sents, trg_sents):
        pred_trg = inference(model, src_sent_tokens, src_sent_synonym_dict, max_len=max_len)
        # cut off </s> token
        pred_trg = pred_trg[:-1]
        trg = trg_sent_tokens[:-1]

        pred_trgs.append(pred_trg)
        trgs.append(trg)

    return bleu_score(pred_trgs, trgs)


if __name__ == '__main__':
    hparams = Hparams()
    args = hparams.args

    token2idx, idx2token = load_vocab(args.vocab)

    # train dataloader
    trainset = GetDataset(args.train_source, args.train_target, args.train_synonym, args.enc_maxlength, args.dec_maxlength, token2idx)
    train_iterator = get_dataloader(args.batch_size, trainset)

    # val dataloader
    valset = GetDataset(args.eval_source, args.eval_target, args.eval_synonym, args.enc_maxlength, args.dec_maxlength, token2idx)
    val_iterator = get_dataloader(args.batch_size, valset, False)

    # train
    train_model(train_iterator, val_iterator, token2idx, idx2token, args, token2idx['<pad>'])

