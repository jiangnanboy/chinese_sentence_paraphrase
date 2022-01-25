from torch.utils.data import Dataset, DataLoader
import torch

class GetDataset(Dataset):
    def __init__(self, source_path, target_path, synonym_path, source_maxlength, target_maxlength, token2idx):
        '''
        :param source_path:
        :param target_path:
        :param synonym_path:
        :param source_maxlength:
        :param target_maxlength:
        :param token2idx:
        '''
        self.source_maxlegnth = source_maxlength
        self.target_maxlength = target_maxlength
        self.token2idx = token2idx
        self.source_sents, self.target_sents, self.synonym_pairs = self.load_data(source_path, target_path, synonym_path)
        self.data_size = len(self.source_sents)

    def load_data(self, source_path, target_path, synonym_path):
        '''
        :param source_path:
        :param target_path:
        :param synonym_path:
        :return:
        '''
        source_sents, target_sents, synonym_pairs = [], [], []
        with open(source_path, 'r', encoding="utf8") as f1, open(target_path, 'r', encoding="utf8") as f2, \
                open(synonym_path, 'r', encoding="utf8") as f3:

            for sent1, sent2, dict_pair in zip(f1, f2, f3):
                if len(sent1.split()) + 1 > self.source_maxlegnth: continue  # 1: </s>
                if len(sent2.split()) + 1 > self.target_maxlength: continue  # 1: </s>
                source_sents.append(sent1.strip())
                target_sents.append(sent2.strip())
                synonym_pairs.append(dict_pair.strip())

        return source_sents, target_sents, synonym_pairs

    def __getitem__(self, idx):
        source_sent = self.source_sents[idx]
        target_sent = self.target_sents[idx]
        synonym_pair = self.synonym_pairs[idx]

        in_words = source_sent.split() + ["</s>"]
        x = [self.token2idx.get(t, self.token2idx["<unk>"]) for t in in_words]
        y = [self.token2idx.get(t, self.token2idx["<unk>"]) for t in ["<s>"] + target_sent.split() + ["</s>"]]
        decoder_input, y = y[:-1], y[1:]  # decoder_input except </s>, y except <s>
        x_paraphrase_dict = []
        word_set, pos_set = set(), set()
        for t in synonym_pair.split():
            tem1, tem2 = t.split("->")  # tem1 -> synonym [同义词对应句子中的词的位置index]
            # (2,3,4,5,...,) -> position index
            pos_set.add(int(tem2 if tem2 != "<unk>" else 0))
            # [[2,3], [4,6], ..., [7,3]]，词典中的同义词与句子中的同义词对应，为index
            # 第一个是词典中的同义词index，第二个是句子中的同义词的position index，会有多个词典中的同义词对应句子中的一个同义词
            x_paraphrase_dict.append([self.token2idx.get(tem1, self.token2idx["<unk>"]), int(tem2 if tem2 != "<unk>" else 0)])
        # source sentence words -> synonym (type: bool)，句子中的词是否是同义词[True, False, True, ...]
        synonym_label = [i in pos_set for i, _ in enumerate(in_words)]

        # padding
        if len(x) < self.source_maxlegnth:
            padding_size = self.source_maxlegnth - len(x)
            x.extend([self.token2idx['<pad>']] * padding_size)
            synonym_label.extend([False] * padding_size)

        if len(decoder_input) < self.target_maxlength:
            decoder_input.extend([self.token2idx['<pad>']] * (self.target_maxlength - len(decoder_input)))
            y.extend([self.token2idx['<pad>']] * (self.target_maxlength - len(y)))

        if len(x_paraphrase_dict) < self.source_maxlegnth:
            x_paraphrase_dict.extend([[self.token2idx['<pad>'], self.token2idx['<pad>']]] * (self.source_maxlegnth - len(x_paraphrase_dict)))

        encoder_input = torch.tensor(x)
        decoder_input = torch.tensor(decoder_input)
        target = torch.tensor(y)
        synonym_dict = torch.tensor(x_paraphrase_dict)
        synonym_label = torch.tensor(synonym_label).long() # bool -> long

        return {'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'target': target,
                'synonym_dict': synonym_dict,
                'synonym_label': synonym_label}

    def __len__(self):
        return self.data_size

def get_dataloader(batch_size, dataset, shuffle=True):
    '''
    :param batch_size:
    :param dataset:
    :param shuffle:
    :return:
    '''
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

def get_train_val_dataloader(batch_size, trainset, train_ratio):
    '''
    split trainset to train and val
    :param batch_size:
    :param trainset:
    :param train_ratio
    :return:
    '''

    train_size = int(train_ratio * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    valloader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False)

    return trainloader, valloader, train_dataset, val_dataset
