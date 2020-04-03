import numpy as np
import string
import argparse
import time
import random
import math

import torch
import torch.nn as nn


from model import *
from preprocess import *
from utility import *
from evaluation import *


# pytorch tesonr: #example * #features
class NeuralNetwork(nn.Module):
    def __init__(self, num_tag_embedding, num_word_embedding, option=1, pretrained_word_embedding=None, word_embedding_dim=300, tag_embedding_dim=5, hidden_dim=8, num_lstm_layer=1):
        super(NeuralNetwork, self).__init__()
        # define all the layers, parameters, etc.

        if option not in [1, 2, 3]:
            print("Option should be 1 or 2 or 3, using default option...")
            option = 1

        if option == 1:
            self.word_embedding = nn.Embedding(num_word_embedding, word_embedding_dim)
            self.fc1 = nn.Linear(word_embedding_dim * 2 + tag_embedding_dim, 50)  # #in_features = word_embedding_dim, #out_features = 50
            self.fc2 = nn.Linear(50, 4)

        elif option == 2:
            weight = torch.FloatTensor(pretrained_word_embedding)
            self.word_embedding = nn.Embedding.from_pretrained(weight)
            self.fc1 = nn.Linear(word_embedding_dim * 2 + tag_embedding_dim, 50)  # #in_features = word_embedding_dim, #out_features = 50
            self.fc2 = nn.Linear(50, 4)

        # Do biLSTM here
        elif option == 3:
            if hidden_dim % 2 != 0:
                hidden_dim += 1
            weight = torch.FloatTensor(pretrained_word_embedding)
            self.word_embedding = nn.Embedding.from_pretrained(weight)
            self.lstm = nn.LSTM(word_embedding_dim, hidden_dim // 2, num_layers=num_lstm_layer, bidirectional=True)
            # self.lstm = nn.LSTM(word_embedding_dim * 2 + tag_embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
            # self.lstm = nn.LSTM(word_embedding_dim + tag_embedding_dim, hidden_dim // 2, num_layers=1,
            #                     bidirectional=True)
            self.fc2 = nn.Linear(hidden_dim, 50)  # #in_features = word_embedding_dim, #out_features = 50

        self.act1 = nn.Sigmoid()
        self.act2 = nn.LogSoftmax(dim=1)
        self.tag_embedding = nn.Embedding(num_tag_embedding, tag_embedding_dim)
        self.option = option

    def forward(self, train_current_words, train_previous_words, train_previous_tags):
        # your model will have two inputs: the current word and the previous tag
        # For the option of training your own embedding,
        # you will need to embed the word and the tag using an embedding layer for each
        # look into nn.Embedding for how to do this
        # you can then concatenate them and feed them through the rest of the network to predict the current tag

        # input of embedding should be torch.LongTensor
        train_current_words = self.word_embedding(train_current_words)
        train_previous_words = self.word_embedding(train_previous_words)
        train_previous_tags = self.tag_embedding(train_previous_tags)
        if self.option == 3:
            x = train_current_words
            # x = torch.cat((train_current_words, train_previous_words, train_previous_tags), 1)
            # x = torch.cat((train_current_words, train_previous_tags), 1)
            h1, _ = self.lstm(x.view(len(train_current_words), 1, -1))
            a2 = self.fc2(h1.view(len(train_current_words), -1))
        else:
            x = torch.cat((train_current_words, train_previous_words, train_previous_tags), 1)
            a1 = self.fc1(x)
            h1 = self.act1(a1)
            a2 = self.fc2(h1)

        y = self.act2(a2)
        return y


# DMEMM with NN and Viterbi
# input words should be list of word indices corresponding to real words in word dict, type: LongTensor
# output tags should be list of tag indices corresponding to real tags in tag dict, type: LongTensor
class DMEMM:
    def __init__(self, nn_model, words, word2idx, tag2idx, START_WORD, START_TAG):
        self.nn = nn_model
        self.words = words
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.START_WORD = START_WORD
        self.START_TAG = START_TAG
        self.sequence_length = len(words)
        self.num_tag_class = len(tag2idx) - 1  # Exclude tag starter
        self.score_matrix = torch.zeros(self.sequence_length, self.num_tag_class)
        self.bp_matrix = torch.LongTensor(self.sequence_length, self.num_tag_class) * 0  # matrix storing back pointers

    def score_sentence(self):
        self.sequence_length
        self.num_tag_class
        # state
        # init
        current_word = self.words[0:1]
        y_hat = math.e**self.nn(current_word, torch.LongTensor([self.word2idx[self.START_WORD]]), torch.LongTensor([self.tag2idx[self.START_TAG]]))
        for j in range(self.num_tag_class):
            self.score_matrix[0][j] = y_hat[0][j]

        # transfer
        for i in range(1, self.sequence_length):
            current_word = self.words[i:i+1]
            previous_word = self.words[i-1:i]
            # row: previous state, column: current state
            transitions = torch.zeros(self.num_tag_class, self.num_tag_class)
            for k in range(self.num_tag_class):
                y_hat = math.e ** self.nn(current_word, previous_word, torch.LongTensor([k]))
                for j in range(self.num_tag_class):
                    transitions[k][j] = y_hat[0][j] * self.score_matrix[i-1][k]
            for j in range(self.num_tag_class):
                self.score_matrix[i][j] = torch.max(transitions[:, j])
                self.bp_matrix[i][j] = torch.argmax(transitions[:, j])
            # print(transitions)
        # print(self.score_matrix)
        # print(self.bp_matrix)

    def viterbi_decode(self):
        tags_predicted = []
        bp = torch.argmax(self.score_matrix[-1, :])
        idx = self.sequence_length
        while idx > 0:
            tags_predicted.append(bp.item())
            idx -= 1
            bp = self.bp_matrix[idx][bp]
        tags_predicted.reverse()
        return tags_predicted

