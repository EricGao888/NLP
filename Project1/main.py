# Issue
# 0. Do cross validation
# 1. Use sklearn to check learing accuracy against training accuracy.
# 2. Do feature engineering for validation data
# 3. Use Embedding
# 4. Do not import tune
# 5. Try stochastic gradient descent instead of batch gradient descent
# 7. Try lem instead of stemming
# 8. Remove not words
# 9. Use Glove instead of Gensim

from sklearn.metrics import f1_score

import os
import sys
import re
import csv
import string
import math
import time

import pandas as pd
import numpy as np
import nltk
from functools import reduce
from random import randrange
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from utility import *
from test import *
from preprocess import *
from model import *

np.random.seed(seed=0)

if __name__ == '__main__':
    start = time.time()

    # Read training and test data
    trainSetDf = readDataFrame(sys.argv[1])
    testSetDf = readDataFrame(sys.argv[2])

    # Preprocess data
    idsTrain, featuresTrain, labelsTrain, idsTest, featuresTest, labelsTest = preprocess(trainSetDf, testSetDf)

    # Use logistic regression
    print("Running logistic regression model...")
    lr = LogisticRegression()
    lr.fit(featuresTrain, labelsTrain, learningRate=0.2, epoch=50000, lam=0)
    labelsPredictedTrain = lr.predict(featuresTrain)
    labelsPredictedTest = lr.predict(featuresTest)
    # print(labelsPredictedTest)

    # Save result for lr
    print('Saving result for logistic regression model...')
    saveResult(testSetDf.copy(), idsTest, labelsPredictedTest, outputFilePath='test_lg.csv')

    # Use neural network
    print("Running neural network model...")
    nn = NeuralNetwork()
    nn.fit(featuresTrain, labelsTrain, learningRate=0.1, epoch=50000, batchSize=1, lam=0, hiddenLayerSize=10)
    labelsPredictedTrain = nn.predict(featuresTrain)
    labelsPredictedTest = nn.predict(featuresTest)
    # print(labelsPredictedTest)

    # Save result for nn
    print('Saving result for neural network model...')
    saveResult(testSetDf.copy(), idsTest, labelsPredictedTest, outputFilePath='test_nn.csv')

    # Check distribution of labels ???

    # Compute running time for script
    end = time.time()
    computeTime(start, end)


