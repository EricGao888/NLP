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


class CrossValidation:
    def __init__(self, rawDf, foldsNum=10, modelType='lr', paramsDict={}):
        self.modelType = modelType
        self.foldsNum = foldsNum
        self.rawDf = rawDf
        self.paramsDict = paramsDict

    def run(self):
        modelType = self.modelType
        foldsNum = self.foldsNum
        rawDf = self.rawDf
        paramsDict = self.paramsDict

        m = rawDf.shape[0]
        unitSize = m / foldsNum

        accuraciesTrain = []
        accuraciesValid = []
        f1sTrain = []
        f1sValid = []
        columns = []

        for i in range(foldsNum):
            print("Runing %d fold cross validation on %dth fold..." % (foldsNum, i+1))
            columns.append('fold' + str(i+1))

            rawDfCopy = rawDf.copy()
            if i == foldsNum - 1:
                validSetDf = rawDf.loc[unitSize*i:-1, :]
            else:
                validSetDf = rawDf.loc[unitSize*i:unitSize*(i+1), :]

            trainSetDf = rawDfCopy.drop(validSetDf.index)
            validSetDf.reset_index(drop=True)
            trainSetDf.reset_index(drop=True)
            idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)

            if modelType == 'lr':
                lr = LogisticRegression()
                lr.fit(featuresTrain, labelsTrain, learningRate=0.01, epoch=23500, lam=0.01)
                labelsPredictedTrain = lr.predict(featuresTrain)
                labelsPredictedValid = lr.predict(featuresValid)
                accuracyTrain = eval.computeAccuracy(labelsPredictedTrain, labelsTrain)
                accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid)
                macroF1Train = f1_score(labelsTrain, labelsPredictedTrain, average='macro')  # Check with sklearn
                macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn
                accuraciesTrain.append(accuracyTrain)
                accuraciesValid.append(accuracyValid)
                f1sTrain.append(macroF1Train)
                f1sValid.append(macroF1Valid)

            if modelType == 'nn':
                nn = NeuralNetwork()
                nn.fit(featuresTrain, labelsTrain, learningRate=0.005, epoch=10000, batchSize=1, lam=0, hiddenLayerSize=2)
                labelsPredictedTrain = nn.predict(featuresTrain)
                labelsPredictedValid = nn.predict(featuresValid)
                accuracyTrain = eval.computeAccuracy(labelsPredictedTrain, labelsTrain)
                accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid)
                macroF1Train = f1_score(labelsTrain, labelsPredictedTrain, average='macro')  # Check with sklearn
                macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn
                accuraciesTrain.append(accuracyTrain)
                accuraciesValid.append(accuracyValid)
                f1sTrain.append(macroF1Train)
                f1sValid.append(macroF1Valid)




def runBaseline(featuresTrain, featuresValid, labelsTrain, labelsValid):
    # Fit model and predict
    bl = Baseline()
    bl.fit(featuresTrain, labelsTrain)
    labelsPredictedValid = bl.predict(featuresValid)

    # Evaluation
    eval = Evaluation()
    accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid)
    # macroF1Valid = eval.computeMacroF1(labelsPredictedValid, labelsValid)
    macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn

    # Print accuracy and macro-f1
    print("Accuracy for Baseline: %.2f%%" % (accuracyValid * 100))
    print("Macro-F1 Score for Baseline: %.2f / 100" % (macroF1Valid * 100))


def tuneLogisticRegression(featuresTrain, featuresValid, labelsTrain, labelsValid):
    # Fit model and predict
    lr = LogisticRegression()
    lr.fit(featuresTrain, labelsTrain, learningRate=0.01, epoch=23500, lam=0.01)
    labelsPredictedTrain = lr.predict(featuresTrain)
    labelsPredictedValid = lr.predict(featuresValid)

    # Evaluation
    eval = Evaluation()
    accuracyTrain = eval.computeAccuracy(labelsPredictedTrain, labelsTrain)
    accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid)
    # macroF1Train = eval.computeMacroF1(labelsPredictedTrain, labelsTrain)
    # macroF1Valid = eval.computeMacroF1(labelsPredictedValid, labelsValid)
    macroF1Train = f1_score(labelsTrain, labelsPredictedTrain, average='macro')  # Check with sklearn
    macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn

    # Print accuracy and macro-f1
    print("Accuracy for Logistic Regression on Training Set: %.2f%%" % (accuracyTrain * 100))
    print("Accuracy for Logistic Regression on Valid Set: %.2f%%" % (accuracyValid * 100))
    print("Macro-F1 Score for Logistic Regression on Training Set: %.2f / 100" % (macroF1Train * 100))
    print("Macro-F1 Score for Logistic Regression on Validation Set: %.2f / 100" % (macroF1Valid * 100))


def tuneNeuralNetwork(featuresTrain, featuresValid, labelsTrain, labelsValid):
    # Fit model and predict
    nn = NeuralNetwork()
    nn.fit(featuresTrain, labelsTrain, learningRate=0.005, epoch=10000, batchSize=1, lam=0, hiddenLayerSize=2)
    labelsPredictedTrain = nn.predict(featuresTrain)
    labelsPredictedValid = nn.predict(featuresValid)

    # Evaluation
    eval = Evaluation()
    accuracyTrain = eval.computeAccuracy(labelsPredictedTrain, labelsTrain)
    accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid)
    # macroF1Train = eval.computeMacroF1(labelsPredictedTrain, labelsTrain)
    # macroF1Valid = eval.computeMacroF1(labelsPredictedValid, labelsValid)
    macroF1Train = f1_score(labelsTrain, labelsPredictedTrain, average='macro')  # Check with sklearn
    macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn

    print("Accuracy for Neural Network on Training Set: %.2f%%" % (accuracyTrain * 100))
    print("Accuracy for Neural Network on Valid Set: %.2f%%" % (accuracyValid * 100))
    print("Macro-F1 Score for Neural Network on Training Set: %.2f / 100" % (macroF1Train * 100))
    print("Macro-F1 Score for Neural Network on Validation Set: %.2f / 100" % (macroF1Valid * 100))

    print(labelsPredictedValid)
    print(labelsValid)


if __name__ == '__main__':
    start = time.time()
    rawDf = readFile(sys.argv[1])
    trainSetDf, validSetDf = sample(rawDf, random_state=1)
    idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)

    # Baseline
    runBaseline(featuresTrain, featuresValid, labelsTrain, labelsValid)

    # Logistic regression
    # tuneLogisticRegression(featuresTrain, featuresValid, labelsTrain, labelsValid)

    # Neural Network
    # tuneNeuralNetwork(featuresTrain, featuresValid, labelsTrain, labelsValid)

    end = time.time()
    computeTime(start, end)