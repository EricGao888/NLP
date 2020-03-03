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

from random import seed, randint
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
        # Z = W(k * n)X(n * m): k * m
        Z -= np.max(Z)  # Avoid sum being too large
        A = np.exp(Z).T / np.sum(np.exp(Z).T, axis=0)  # A: m * k
        return A

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
            if i % 500 == 0:
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

class Baseline:
    def __init__(self):
        self.labels = None

    def fit(self, X, Y):
        self.labels = list(set(Y.flatten()))
        self.labels.sort()

    def predict(self, X):
        m = X.shape[0]
        upper = len(self.labels) - 1
        seed(1)
        YPredicted = np.zeros((m, 1))
        for i in range(m):
            YPredicted[i, 0] = randint(0, upper)
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
    def __init__(self, trainSetDf, foldsNum, modelType, paramsDict):
        self.modelName = modelName
        self.foldsNum = foldsNum
        self.trainSetDf = trainSetDf
        self.paramsDict = paramsDict

    def run(self):
        pass


class NeuralNetwork:
    def __init__(self):
        self.Wh = None  # Weight matrix of hidden layer
        self.Bh = None  # Bias term matrix of hidden layer
        self.Wo = None  # Weight matrix of output layer
        self.Bo = None  # Bias term matrix of output layer

        self.m = 0  # Number of training examples
        self.n = 0  # Number of input features
        self.k = 0  # Number of distinct classes
        self.learningRate = 0
        self.lamb = 0
        self.hiddenLayerSize = 0
        self.labelEncodingDict = {}
        self.labelDecodingDict = {}

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def softmax(self, Z):
        # Z = W(k * n)X(n * m): k * m
        Z -= np.max(Z)  # Avoid sum being too large
        A = np.exp(Z).T / np.sum(np.exp(Z).T, axis=0)  # A: m * k
        return A

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

    def fit(self, X, Y, learningRate=0.005, epoch=9430, batchSize=1, lam=0, hiddenLayerSize=10):
        # Get one-hot encoded Y
        YOneHot = self.oneHotEncode(Y)  # Y: m * 1, YOneHot: m * k
        m = X.shape[0]  # Retrieve the number of examples
        k = np.unique(Y).shape[0]  # Retrieve the number of distinct classes
        Bh = np.random.randn(hiddenLayerSize, 1)
        Bo = np.random.randn(k, 1)
        n = X.shape[1]  # Retrieve the number of features
        X = X.T  # Transpose X to n * m (one example for one column)
        Wh = np.random.rand(hiddenLayerSize, n)  # Wh: hiddenLayerSize * n
        # Wh(hiddenLayerSize * n)X(n * m) = Xh: hiddenLayerSize * m
        Wo = np.random.rand(k, hiddenLayerSize)  # Wo: k * hiddenLayerSize
        # Wo(k * hiddenLayerSize)Xh(hiddenLayerSize * m) = Xo: k * m
        # print("m: %d, n: %d, k: %d" % (m, n, k))
        self.m = m
        self.n = n
        self.k = k
        self.hiddenLayerSize = hiddenLayerSize

        for i in range(1, epoch + 1):
            # Feed Forward Computation

            ## Input Layer ---> Hidden Layer
            Zh = np.dot(Wh, X) + Bh  # Wh(hiddenLayerSize * n)X(n * m) = Zh: hiddenLayerSize * m
            Ah = self.sigmoid(Zh)  # Ah: hiddenLayerSize * m

            ## Hidden Layer ---> Output Layer
            Zo = np.dot(Wo, Ah) + Bo  # Wo(k * hiddenLayerSize)Ah(hiddenLayerSize * m) = Zo: k * m
            Ao = self.softmax(Zo)  # Ao: m * k, softmax performs transposition on return value

            # Backpropagation

            ## Output Layer ---> Hidden Layer
            dCostDZo = Ao - YOneHot  # YOneHot: m * k, Ao: m * k, dCostDZo: m * k
            dZoDWo = Ah  # dZoDWo = Ah: hiddenLayerSize * m
            dCostDWo = np.dot(dZoDWo, dCostDZo)  # dCostDWo: hiddenLayerSize * k
            dCostDBo = dCostDZo  # dCostDBo = m * k

            ## Hidden Layer ---> Input Layer
            dZoDAh = Wo  # dZoDAh: k * hiddenLayerSize
            dCostDAh = np.dot(dCostDZo, dZoDAh)  # dCostDAh: m * hiddenLayerSize
            dAhDZh = self.sigmoid(Zh) * (1 - self.sigmoid(Zh))  # dAhDZh: hiddenLayerSize * m
            dZhDWh = X  # dZhDWh: n * m
            dCostDWh = np.dot(dZhDWh, dAhDZh.T * dCostDAh)  # dCostDWh: n * hiddenLayerSize
            dCostDBh = dCostDAh * dAhDZh.T  # dCostDBh: m * hiddenLayerSize

            # Update
            Wh -= (1 / m) * learningRate * dCostDWh.T  # dCostDWh.T: hiddenLayerSize * n
            Bh -= (1 / m) * learningRate * np.reshape(np.sum(dCostDBh, axis=0), (-1, 1))  # dCostDBh: m * hiddenLayerSize
            Wo -= (1 / m) * learningRate * dCostDWo.T  # dCostDWh.T: k * hiddenLayerSize
            Bo -= (1 / m) * learningRate * np.reshape(np.sum(dCostDBo.T, axis=1), (-1, 1))  # dCostDBo.T = k * m

            if i % 100 == 0:
                cost = (1 / m) * np.sum(-YOneHot * np.log(Ao))
                print("Cost = %f in %dth Epoch..." % (cost, i))

        self.Wh = Wh
        self.Bh = Bh
        self.Wo = Wo
        self.Bo = Bo

    def predict(self, X):
        Wh = self.Wh
        Wo = self.Wo
        Bh = self.Bh
        Bo = self.Bo
        Zh = np.dot(Wh, X.T) + Bh
        Ah = self.sigmoid(Zh)  # Ah: hiddenLayerSize * m
        Zo = np.dot(Wo, Ah) + Bo  # Wo(k * hiddenLayerSize)Ah(hiddenLayerSize * m) = Zo: k * m
        Ao = self.softmax(Zo)  # Ao: m * k, softmax performs transposition on return value
        Y = np.argmax(Ao, axis=1)
        YPredicted = pd.DataFrame(Y, columns=['label'])['label'].apply(lambda x: self.decode(x)).values
        YPredicted = np.reshape(YPredicted, (-1, 1))
        return YPredicted


