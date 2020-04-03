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


# function to train the model and keep track of the losses
def train_epoch(model, opt, criterion, train_current_words, train_previous_words, train_previous_tags, train_gold_labels, batch_size=1000):
    model.train()
    losses = []
    for beg_i in range(0, len(train_current_words), batch_size):
        train_current_words_batch = Variable(torch.LongTensor(train_current_words[beg_i:beg_i + batch_size]))
        train_previous_words_batch = Variable(torch.LongTensor(train_previous_words[beg_i:beg_i + batch_size]))
        train_previous_tags_batch = Variable(torch.LongTensor(train_previous_tags[beg_i:beg_i + batch_size]))
        train_gold_labels_batch = Variable(torch.LongTensor(train_gold_labels[beg_i:beg_i + batch_size]))

        opt.zero_grad()
        # (1) Forward
        y_hat = model(train_current_words_batch, train_previous_words_batch, train_previous_tags_batch)
        # (2) Compute diff
        loss = criterion(y_hat, train_gold_labels_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return [sum(losses)/float(len(losses))]

def tune_nn_batch():
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
    pretrained_word_embedding = get_pretrained_embedding(word2idx, idx2word) if option in [2, 3] else None

    # Tune NN
    batch_sizes = [50, 100, 500, 1000, 5000, 10000]
    train_accuracies = []
    test_accuracies = []

    for batch_size in batch_sizes:
        net = NeuralNetwork(len(tag2idx), len(word2idx), option=option,
                            pretrained_word_embedding=pretrained_word_embedding,
                            word_embedding_dim=300, tag_embedding_dim=5, hidden_dim=8, num_lstm_layer=1)

        opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()
        num_epochs = 10

        for e in range(num_epochs):
            print('batch size: %d, epoch: %d' % (batch_size, e + 1))
            train_epoch(net, opt, criterion, train_current_words, train_previous_words, train_previous_tags,
                        train_gold_labels, batch_size=batch_size)

        train_current_words = torch.LongTensor(train_current_words)
        train_previous_words = torch.LongTensor(train_previous_words)
        train_previous_tags = torch.LongTensor(train_previous_tags)
        test_current_words = torch.LongTensor(test_current_words)
        test_previous_words = torch.LongTensor(test_previous_words)
        test_previous_tags = torch.LongTensor(test_previous_tags)

        train_y_hat = torch.argmax(net(train_current_words, train_previous_words, train_previous_tags), dim=1)
        test_y_hat = torch.argmax(net(test_current_words, test_previous_words, test_previous_tags), dim=1)

        train_accuracy = compute_accuracy(train_y_hat, train_gold_labels)
        test_accuracy = compute_accuracy(test_y_hat, test_gold_labels)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    fig, ax = plt.subplots()
    ax.plot(batch_sizes, train_accuracies, label="Train", marker='o', markersize=2)
    ax.plot(batch_sizes, test_accuracies, label="Test", marker='o', markersize=2)
    ax.set(xlabel='Batch Size', ylabel='Accuracy',
           title='Accuracy against Batch Size')
    ax.set_ylim(0, max(train_accuracies + test_accuracies) * 1.1)
    ax.set_xlim(0, max(batch_sizes) * 1.1)
    ax.grid()
    ax.legend()
    fig.savefig("Batch.png")



def tune_nn_epoch():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--option', type=int, default=1,
                        help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    args, _ = parser.parse_known_args()
    option = args.option

    # Read dataset
    train_set, valid_set = sample_data(read_data(path=args.train_file))
    test_set = read_data(path=args.test_file)

    # Preprocess
    word2idx, idx2word, tag2idx, idx2tag = initialize()
    train_current_words, train_previous_words, train_previous_tags, train_gold_labels = prepare_train_data(train_set,
                                                                                                           word2idx,
                                                                                                           idx2word,
                                                                                                           tag2idx)
    valid_current_words, valid_previous_words, valid_previous_tags, valid_gold_labels = prepare_test_data(valid_set,
                                                                                                      word2idx, tag2idx)
    test_current_words, test_previous_words, test_previous_tags, test_gold_labels = prepare_test_data(test_set,
                                                                                                      word2idx, tag2idx)
    # Tune NN
    batch_size = 1000

    train_accuracies = []
    valid_accuracies = []
    test_accuracies = []
    pretrained_word_embedding = get_pretrained_embedding(word2idx, idx2word) if option in [2, 3] else None
    net = NeuralNetwork(len(tag2idx), len(word2idx), option=option,
                        pretrained_word_embedding=pretrained_word_embedding,
                        word_embedding_dim=300, tag_embedding_dim=5, hidden_dim=8, num_lstm_layer=1)

    opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
    criterion = nn.NLLLoss()
    num_epochs = [x for x in range(1, 101)]

    for e in num_epochs:
        print('option: %d, epoch: %d' % (option, e))
        train_epoch(net, opt, criterion, train_current_words, train_previous_words, train_previous_tags,
                    train_gold_labels, batch_size=batch_size)

        train_current_words = torch.LongTensor(train_current_words)
        train_previous_words = torch.LongTensor(train_previous_words)
        train_previous_tags = torch.LongTensor(train_previous_tags)
        valid_current_words = torch.LongTensor(valid_current_words)
        valid_previous_words = torch.LongTensor(valid_previous_words)
        valid_previous_tags = torch.LongTensor(valid_previous_tags)
        test_current_words = torch.LongTensor(test_current_words)
        test_previous_words = torch.LongTensor(test_previous_words)
        test_previous_tags = torch.LongTensor(test_previous_tags)

        train_y_hat = torch.argmax(net(train_current_words, train_previous_words, train_previous_tags), dim=1)
        valid_y_hat = torch.argmax(net(valid_current_words, valid_previous_words, valid_previous_tags), dim=1)
        test_y_hat = torch.argmax(net(test_current_words, test_previous_words, test_previous_tags), dim=1)

        train_accuracy = compute_accuracy(train_y_hat, train_gold_labels)
        valid_accuracy = compute_accuracy(valid_y_hat, valid_gold_labels)
        test_accuracy = compute_accuracy(test_y_hat, test_gold_labels)
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        test_accuracies.append(test_accuracy)

    fig, ax = plt.subplots()
    ax.plot(num_epochs, train_accuracies, label="Train", marker='o', markersize=2)
    ax.plot(num_epochs, valid_accuracies, label="Valid", marker='o', markersize=2)
    ax.plot(num_epochs, test_accuracies, label="Test", marker='o', markersize=2)
    ax.set(xlabel='Epoch', ylabel='Accuracy',
           title='Accuracy against Epoch')
    ax.set_ylim(0, max(train_accuracies + valid_accuracies + test_accuracies) * 1.1)
    ax.set_xlim(0, max(num_epochs) * 1.1)
    ax.grid()
    ax.legend()
    fig.savefig("epoch.png")


def compare_three_options():
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

    for option in [1, 2, 3]:
        train_accuracies = []
        test_accuracies = []
        pretrained_word_embedding = get_pretrained_embedding(word2idx, idx2word) if option in [2, 3] else None
        net = NeuralNetwork(len(tag2idx), len(word2idx), option=option,
                            pretrained_word_embedding=pretrained_word_embedding,
                            word_embedding_dim=300, tag_embedding_dim=5, hidden_dim=8, num_lstm_layer=1)

        opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()
        num_epochs = [x for x in range(1, 101)]

        for e in num_epochs:
            print('option: %d, epoch: %d' % (option, e))
            train_epoch(net, opt, criterion, train_current_words, train_previous_words, train_previous_tags,
                        train_gold_labels, batch_size=batch_size)

            train_current_words = torch.LongTensor(train_current_words)
            train_previous_words = torch.LongTensor(train_previous_words)
            train_previous_tags = torch.LongTensor(train_previous_tags)
            test_current_words = torch.LongTensor(test_current_words)
            test_previous_words = torch.LongTensor(test_previous_words)
            test_previous_tags = torch.LongTensor(test_previous_tags)

            train_y_hat = torch.argmax(net(train_current_words, train_previous_words, train_previous_tags), dim=1)
            test_y_hat = torch.argmax(net(test_current_words, test_previous_words, test_previous_tags), dim=1)

            train_accuracy = compute_accuracy(train_y_hat, train_gold_labels)
            test_accuracy = compute_accuracy(test_y_hat, test_gold_labels)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

        fig, ax = plt.subplots()
        ax.plot(num_epochs, train_accuracies, label="Train", marker='o', markersize=2)
        ax.plot(num_epochs, test_accuracies, label="Test", marker='o', markersize=2)
        ax.set(xlabel='Epoch', ylabel='Accuracy',
               title='Accuracy against Epoch for Option %d' % option)
        ax.set_ylim(0, max(train_accuracies + test_accuracies) * 1.1)
        ax.set_xlim(0, max(num_epochs) * 1.1)
        ax.grid()
        ax.legend()
        fig.savefig("option%d.png" % option)

        # Test DMEMM
        TP = 0
        FP = 0
        FN = 0
        print_cnt = 0
        for example in test_set:
            words = torch.LongTensor([word2idx[x] if x in word2idx else random.randint(1, len(word2idx) - 1) for x in example['words']])
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


def main():
    start = time.time()
    random.seed(0)
    # tune_nn_batch()
    tune_nn_epoch()
    # compare_three_options()
    # Compute time cost
    end = time.time()
    compute_time(start, end)

if __name__ == '__main__':
    main()

