import numpy as np
import string

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import random
import math


# define 'constant'
START_WORD = '$$$word_starter$$$'
START_TAG = '$$$tag_starter$$$'
NUM_CLASS = 4


def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')

                if tag == 'O':
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset


def sample_data(train_set, seed=0):
    random.seed(seed)
    train_indices = [x for x in range(len(train_set))]
    valid_indices = []
    while len(train_indices) > len(train_set) * 0.8:
        idx = random.randint(0, len(train_indices) - 1)
        valid_indices.append(train_indices[idx])
        del train_indices[idx]
    train_set_new = [train_set[x] for x in train_indices]
    valid_set = [train_set[x] for x in valid_indices]
    print(train_indices)
    print(valid_indices)
    return train_set_new, valid_set


def initialize():
    # Mapping for indices of embedding, real tag index must start from 0 to stay consistent with softmax output
    # Index of word starter can start from zero since it will not miss with output
    # Only 4 classes, tag starter index only used for embedding
    word2idx = {START_WORD: 0}
    idx2word = {0: START_WORD}
    tag2idx = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3, START_TAG: 4}
    idx2tag = {0: 'O', 1: 'T-POS', 2: 'T-NEG', 3: 'T-NEU'}
    return word2idx, idx2word, tag2idx, idx2tag


def prepare_train_data(train_set, word2idx, idx2word, tag2idx):
    train_current_words = []
    train_previous_words = []
    train_previous_tags = []
    train_gold_labels = []
    # Construct indices dict and input
    # word_i, tag_i-1 => tag_i
    cnt = 1  # include tag starter
    for example in train_set:
        words = example['words']
        tags = example['ts_raw_tags']
        word_idx = 0
        tag_idx = 0
        for word in words:
            if word not in word2idx:
                word2idx[word] = cnt
                idx2word[cnt] = word
                cnt += 1
            if word_idx == 0:
                train_previous_words.append(word2idx[START_WORD])
            else:
                train_previous_words.append(train_current_words[-1])
            train_current_words.append(word2idx[word])
            word_idx += 1
        for tag in tags:
            if tag_idx == 0:
                train_previous_tags.append(tag2idx[START_TAG])
            else:
                train_previous_tags.append(train_gold_labels[-1])
            train_gold_labels.append(tag2idx[tag])
            tag_idx += 1

    return train_current_words, train_previous_words, train_previous_tags, train_gold_labels


def prepare_test_data(test_set, word2idx, tag2idx):
    test_current_words = []
    test_previous_words = []
    test_previous_tags = []
    test_gold_labels = []
    for example in test_set:
        words = example['words']
        tags = example['ts_raw_tags']
        word_idx = 0
        tag_idx = 0
        for word in words:
            if word_idx == 0:
                test_previous_words.append(word2idx[START_WORD])
            else:
                test_previous_words.append(test_current_words[-1])
            if word in word2idx:
                test_current_words.append(word2idx[word])
            else:
                # Deal with unseen words
                # randint both inclusive, start from 1 to avoid assigned with starter
                test_current_words.append(random.randint(1, len(word2idx) - 1))
            word_idx += 1
        for tag in tags:
            if tag_idx == 0:
                test_previous_tags.append(tag2idx[START_TAG])
            else:
                test_previous_tags.append(test_gold_labels[-1])
            test_gold_labels.append(tag2idx[tag])
            tag_idx += 1

        return test_current_words, test_previous_words, test_previous_tags, test_gold_labels


def get_pretrained_embedding(word2idx, idx2word):
    # note: this will only work on a cs machine (ex: data.cs.purdue.edu)
    wv = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
    # wv = KeyedVectors.load_word2vec_format(datapath("/Users/ericgao/workroot/Data/w2v.bin"), binary=True)
    vocabulary = wv.vocab.keys()
    vocabulary = set(filter(lambda x: x in vocabulary and x not in word2idx, vocabulary))
    # Deal with unseen words in training set while using word2vec
    unseen_embedding = list(np.average(np.array([wv[x] for x in vocabulary]), axis=0))
    pretrained_word_embedding = [wv[idx2word[idx]] if idx2word[idx] in wv else unseen_embedding for idx in range(len(idx2word))]
    return pretrained_word_embedding


