# m: #examples, n: #features (including bias term), k: #classes
# http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
import os
import sys
import re
import csv
import string
import math

import pandas as pd
import numpy as np
import nltk
from functools import reduce
from random import randrange
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

np.random.seed(seed=0)

class LogisticRegression:
    def __init__(self):
        self.W = None
        self.m = 0
        self.n = 0
        self.k = 0
        self.labelEncodingDict = {}
        self.labelDecodingDict = {}

    def computeGradient(self, W, X, YOneHot, lam):
        m = self.m
        Z = np.dot(W, X)  # Z = W(k * n)X(n * m): k * m
        P = self.softmax(Z)  # P: m * k
        # YOneHot(m * k) * softmax(m * k): m * k
        cost = (-1 / m) * np.sum(YOneHot * np.log(P)) + (lam / 2) * np.sum(W * W)  # X(n * m)(YOneHot - P)(m * k): n * k
        gradient = (-1 / m) * np.dot(X, (YOneHot - P)).T + lam * W  # Gradient should be consistent with W as k * n
        return gradient, cost

    def softmax(self, Z):
        Z -= np.max(Z)  # Z = W(k * n)X(n * m): k * m
        res = np.exp(Z).T / np.sum(np.exp(Z).T, axis=0)  # Avoid sum being too large, res: m * k
        return res

    def encode(self, x):
        oneHotList = [0 for i in range(len(self.labelEncodingDict))]
        oneHotList[self.labelEncodingDict[x]] = 1
        return oneHotList

    def decode(self, y):
        return self.labelDecodingDict[y]

    def oneHotEncode(self, Y):
        labels = list(set(Y.flatten()))
        labels.sort()
        self.labelEncodingDict = dict.fromkeys(labels, 0)
        value = 0
        for label in labels:
            self.labelEncodingDict[label] = value
            self.labelDecodingDict[value] = label
            value += 1
        return pd.DataFrame(pd.DataFrame(Y, columns=['label'])['label'].
                            apply(lambda x: self.encode(x)).tolist()).values

    def fit(self, X, Y, learningRate=0.005, epoch=9430, batchSize=1, lam=0):
        # Get one-hot encoded Y
        YOneHot = self.oneHotEncode(Y)  # Y: m * 1, YOneHot: m * k
        # Add bias term to feature matrix
        X = np.insert(X, -1, 1, axis=1)
        m = X.shape[0]  # Retrieve the number of examples
        n = X.shape[1]  # Retrieve the number of features
        k = np.unique(Y).shape[0]  # Retrieve the number of
        X = X.T  # Transpose X to n * m (one example for one column)
        W = np.random.rand(k, n)  # W: k * n
        # print("m: %d, n: %d, k: %d" % (m, n, k))
        self.m = m
        self.n = n
        self.k = k
        for i in range(1, epoch + 1):
            gradient, cost = self.computeGradient(W, X, YOneHot, lam)
            W = W - learningRate * gradient
            if i % 10 == 0:
                print("Cost = %f in %dth Epoch..." % (cost, i))

        print(W)
        self.W = W

    def predict(self, X):
        X = np.insert(X, -1, 1, axis=1)  # Add bias term
        Z = np.dot(self.W, X.T)  # Z = W(k * n)X.T(n * m): k * m
        P = self.softmax(Z)  # P: m * k
        # Decode label after prediction
        Y = np.argmax(P, axis=1)
        YPredicted = pd.DataFrame(Y, columns=['label'])['label'].apply(lambda x: self.decode(x)).values
        YPredicted = np.reshape(YPredicted, (-1, 1))
        return YPredicted


class NeuralNetwork:
    def __init__(self):
        self.W = 0
        self.learningRate = 0
        self.batchSize = 0
        self.lamb = 0

    def fit(self, X, Y, learningRate=0, batchSize=1, lamb=0):
        pass

    def predict(self, X):
        pass


class Baseline:
    def __init__(self):
        self.labels = None

    def fit(self, X, Y):
        self.labels = list(set(Y.flatten()))
        self.labels.sort()

    def predict(self, X):
        np.random.rand
        return YPredicted

class Evaluation:
    def __init__(self):
        pass

    def computeMacroF1(self, YPredicted, YGold):
        if not ((YPredicted.shape[0] == YGold.shape[0]) and (YPredicted.shape[1] == YGold.shape[1])):
            raise ValueError('YPredicted and YGold not in same dimension!')
        m = YPredicted.shape[0]
        labels = list(set(YPredicted.flatten()).union(YGold.flatten()))
        labels.sort()
        k = len(labels)
        confusionMatrix = np.zeros((k, k))
        labelEncodingDict = dict.fromkeys(labels, 0)
        value = 0
        for label in labels:
            labelEncodingDict[label] = value
            value += 1

        f1Dict = {}
        encodedLabelsPredicted = list(pd.DataFrame(YPredicted, columns=['label'])['label']
                                      .apply(lambda x: labelEncodingDict[x]))
        encodedLabelsGold = list(pd.DataFrame(YGold, columns=['label'])['label']
                                 .apply(lambda x: labelEncodingDict[x]))
        for i in range(k):
            f1Dict[i] = -1

        for i in range(m):
            confusionMatrix[encodedLabelsGold[i]][encodedLabelsPredicted[i]] += 1

        for i in range(k):
            if np.sum(confusionMatrix[i]) != 0:
                f1Dict[i] = 2.0 * confusionMatrix[i][i] / (np.sum(confusionMatrix[i]) + np.sum(confusionMatrix[:, i]))
            elif np.sum(confusionMatrix[:, i]) != 0:
                f1Dict[i] = 0
            else:
                m -= 1

        count = 0
        for label in f1Dict:
            if f1Dict[label] != -1:
                count += f1Dict[label]
        return float(count) / m

    def computeAccuracy(self, YPredicted, YGold):
        if not ((YPredicted.shape[0] == YGold.shape[0]) and (YPredicted.shape[1] == YGold.shape[1])):
            raise ValueError('YPredicted and YGold not in same dimension!')
        m = YPredicted.shape[0]
        return np.sum(YPredicted == YGold) / m

class crossValidation:
    def __init__(self, modelName):
        self.modelName = modelName

    def run(self):
        pass


