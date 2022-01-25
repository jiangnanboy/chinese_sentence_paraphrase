
def load_vocab(vocab_fpath):
    '''
    :param vocab_fpath:
    :return:
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    print('vocab len: {}'.format(len(token2idx)))
    return token2idx, idx2token

