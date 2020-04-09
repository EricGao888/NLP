import numpy as np
import string
import argparse
import time
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

from model import *
from preprocess import *
from utility import *
from evaluation import *
from tune import *

def main():
    start = time.time()
    random.seed(0)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--option', type=int, default=1,
                        help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    args, _ = parser.parse_known_args()
    option = args.option

    # Read dataset
    train_set = read_data(path=args.train_file)
    test_set = read_data(path=args.test_file)

    # Preprocess
    word2idx, idx2word, tag2idx, idx2tag = initialize()
    train_current_words, train_previous_words, train_previous_tags, train_gold_labels = prepare_train_data(train_set,
                                                                                                           word2idx,
                                                                                                           idx2word,
                                                                                                           tag2idx)
    test_current_words, test_previous_words, test_previous_tags, test_gold_labels = prepare_test_data(test_set,
                                                                                                      word2idx, tag2idx)
    # Tune NN
    batch_size = 1000

    pretrained_word_embedding = get_pretrained_embedding(word2idx, idx2word) if option in [2, 3] else None
    net = NeuralNetwork(len(tag2idx), len(word2idx), option=option,
                        pretrained_word_embedding=pretrained_word_embedding,
                        word_embedding_dim=300, tag_embedding_dim=5, hidden_dim=8, num_lstm_layer=1)

    opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()
    num_epochs = [x for x in range(1, 51)]

    for e in num_epochs:
        loss = train_epoch(net, opt, criterion, train_current_words, train_previous_words, train_previous_tags,
                    train_gold_labels, batch_size=batch_size)
        print("option: %d, epoch: %d, loss: %f" % (option, e, loss[0]))

    # Test DMEMM
    TP = 0
    FP = 0
    FN = 0
    print_cnt = 0
    for example in test_set:
        words = torch.LongTensor(
            [word2idx[x] if x in word2idx else random.randint(1, len(word2idx) - 1) for x in example['words']])
        tags_gold = example['ts_raw_tags']
        dmemm = DMEMM(net, words, word2idx, tag2idx, START_WORD, START_TAG)
        dmemm.score_sentence()
        tags_predicted = [idx2tag[x] for x in dmemm.viterbi_decode()]
        if len(tags_gold) != len(tags_predicted):
            print("Gold tags and predicted tags not in the same dimension!")
        for pair in zip(tags_gold, tags_predicted):
            if pair[0] != 'O' and pair[1] != 'O':
                if pair[0] == pair[1]:
                    TP += 1
                else:
                    FP += 1
                    FN += 1
            elif pair[0] != 'O':
                FN += 1
            elif pair[1] != 'O':
                FP += 1
        print_cnt += 1
        if print_cnt == 10:
            pass
            # break
    precision, recall, f1 = evaluate(TP, FP, FN, correction=True)
    print("Evaluation ===> precision: %.2f / 100, recall: %.2f / 100, f1: %.2f / 100" % (
        precision * 100, recall * 100, f1 * 100))
    # Compute time cost
    end = time.time()
    compute_time(start, end)

if __name__ == '__main__':
    main()





