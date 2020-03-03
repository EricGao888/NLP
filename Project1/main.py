# Issue
# 0. Do cross validation
# 1. Use sklearn to check learing accuracy against training accuracy.
# 2. Do feature engineering for validation data
# 3. Use Embedding
# 4. Comment sklearn
# 5. Use stochastic gradient descent instead of batch gradient descent

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

from sklearn.metrics import f1_score

np.random.seed(seed=0)

if __name__ == '__main__':
    start = time.time()

    rawDf = readFile(sys.argv[1])
    trainSetDf, validSetDf = sample(rawDf, 1)
    idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)

    # Baseline
    # bl = Baseline()
    # bl.fit(featuresTrain, labelsTrain)
    # labelsPredicted = bl.predict(featuresValid)

    # Logistic regression
    # lr = LogisticRegression()
    # lr.fit(featuresTrain, labelsTrain, learningRate=0.01, epoch=50000)
    # labelsPredictedTrain = lr.predict(featuresTrain)
    # labelsPredictedValid = lr.predict(featuresValid)

    # Neural Network
    nn = NeuralNetwork()
    nn.fit(featuresTrain, labelsTrain, learningRate=0.005, epoch=10000, batchSize=1, lam=0, hiddenLayerSize=2)
    labelsPredictedTrain = nn.predict(featuresTrain)
    labelsPredictedValid = nn.predict(featuresValid)

    # Evaluation
    eval = Evaluation()
    accuracyTrain = eval.computeAccuracy(labelsPredictedTrain, labelsTrain) * 100
    accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid) * 100
    macroF1Train = eval.computeMacroF1(labelsPredictedTrain, labelsTrain) * 100
    macroF1Valid = eval.computeMacroF1(labelsPredictedValid, labelsValid) * 100
    # macroF1Ans = f1_score(labelsValid, labelsPredicted, average='macro') * 100  # Check with sklearn

    # Print result for baseline
    # print("Accuracy for Baseline: %.2f%%" % accuracy)
    # print("Macro-F1 Score for Baseline: %.2f" % macroF1)
    # print("Macro-F1 Score for Baseline from sklean: %.2f" % macroF1)

    # Print result for logistic regression
    # print("Accuracy for Logistic Regression on Training Set: %.2f%%" % accuracyTrain)
    # print("Accuracy for Logistic Regression on Valid Set: %.2f%%" % accuracyValid)
    # print("Macro-F1 Score for Logistic Regression on Training Set: %.2f" % macroF1Train)
    # print("Macro-F1 Score for Logistic Regression on Validation Set: %.2f" % macroF1Valid)
    # print("Macro-F1 Score for Logistic Regression from sklean: %.2f" % macroF1)

    # Print result for neural network
    print("Accuracy for Neural Network on Training Set: %.2f%%" % accuracyTrain)
    print("Accuracy for Neural Network on Valid Set: %.2f%%" % accuracyValid)
    print("Macro-F1 Score for Neural Network on Training Set: %.2f" % macroF1Train)
    print("Macro-F1 Score for Neural Network on Validation Set: %.2f" % macroF1Valid)
    # print("Macro-F1 Score for Logistic Regression from sklean: %.2f" % macroF1)

    # print(labelsPredicted)
    # print(labelsValid)

    # text2BowTest()
    # LR()
    # NN()

    end = time.time()
    computeTime(start, end)


