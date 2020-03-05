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


def readDataFrame(inputFilePath):
    rawDf = pd.read_csv(inputFilePath, header=[0])
    return rawDf


def saveDataFrame(resultDf, outputFilePath):
    if outputFilePath == '':
        resultDf.to_csv('output.csv', index=False)
    else:
        resultDf.to_csv(outputFilePath, index=False)


def saveResult(testSetDf, idsTest, labelsPredictedTest, outputFilePath=''):
    ids = list(idsTest.flatten())
    labels = list(labelsPredictedTest.flatten())
    id2LabelDict = dict(zip(ids, labels))
    testSetDf['label'] = testSetDf.apply(lambda x: id2LabelDict[x.tweet_id], axis=1)
    saveDataFrame(testSetDf, outputFilePath)


def sample(rawDf, random_state=1):
    rawDfCopy = rawDf.copy()
    trainSetDf = rawDf.sample(frac=0.8, random_state=random_state)
    validSetDf = rawDfCopy.drop(trainSetDf.index)
    trainSetDf.reset_index(drop=True)
    validSetDf.reset_index(drop=True)
    return trainSetDf, validSetDf


def computeTime(start, end):
    minutes = int((end - start) / 60)
    seconds = (end - start) % 60
    print("Total time cost: %d minutes %d seconds" % (minutes, seconds))

def computeDistribution(rawDf):
    labels = list(rawDf['label'])
    labelDict = dict.fromkeys(set(labels), 0)
    for label in labels:
        labelDict[label] += 1
    print(labelDict)


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

