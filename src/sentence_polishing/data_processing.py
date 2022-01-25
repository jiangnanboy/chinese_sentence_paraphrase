
from collections import defaultdict
import operator
import os

#处理数据，生成词典vocab，这里需要分词并限制词典大小
class Vocab(object):
    unk = '<unk>'
    sos = '<sos>'
    eos = '<eos>'
    def __init__(self, unk=unk):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)
        self.add_word(self.sos, count=0)
        self.add_word(self.eos, count=0)

    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count
        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print('{} total words with {} uniques'.format(self.total_words, len(self.word_freq)))

    def limit_vocab_length(self, length):
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0

    def save_vocab(self, filePath):
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'w', encoding='utf8') as fd:
            for (word, freq) in sorted_tup:
                fd.write((u'%s\t%d\n'%(word, freq)))
        print('word_freq len: {}'.format(len(self.word_freq)))

    def save_vocab_for_nmt(self, filePath):
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'w', encoding='utf8') as fd:
            fd.write('<unk>\n')
            fd.write('<s>\n')
            fd.write('</s>\n')
            for (word, freq) in sorted_tup:
                fd.write((u'%s\n' % (word)))

    def load_vocab_from_file(self, filePath, sep='\t'):
        with open(filePath, 'r', encoding='utf8') as fd:
            for line in fd:
                word, freq = line.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                self.word_freq[word] = int(freq)
            print('load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq)))

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

    def __len__(self):
        return len(self.word_to_index)

vocab = Vocab()
def read_data(filename):
    with open(filename, 'r', encoding='utf8') as fd:
        line_count = 0
        for line in fd:
            line_count += 1
            line = line.strip()
            if line == '':
                continue
            for word in line.split():
                vocab.add_word(word)
    print('filename: {}, total lines: {}'.format(filename, line_count))

if __name__ == '__main__':
    dir_path = os.path.abspath(os.path.join(os.getcwd(), '../../data/sentence_polishing'))

    vocab_path = os.path.join(dir_path, 'vocab.vocab')
    vocab_gaixie = os.path.join(dir_path, 'vocab.gaixie')

    train_src_path = os.path.join(dir_path, 'train.src')
    train_tgt_path = os.path.join(dir_path, 'train.tgt')
    test_src_path = os.path.join(dir_path, 'test.src')
    test_tgt_path = os.path.join(dir_path, 'test.tgt')
    dev_src_path = os.path.join(dir_path, 'dev.src')
    dev_tgt_path = os.path.join(dir_path, 'dev.tgt')

    # word and freq
    read_data(train_src_path)
    read_data(train_tgt_path)
    vocab.save_vocab(vocab_gaixie)

    vocab = Vocab()
    vocab.load_vocab_from_file(vocab_gaixie)
    vocab.limit_vocab_length(32000) # 限制了词典的大小
    vocab.save_vocab_for_nmt(vocab_path)

