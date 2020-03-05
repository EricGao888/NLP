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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        index = ['Train Accuracy', 'Validation Accuracy', 'Train F1', 'Validation F1']

        for i in range(foldsNum):
            print("Runing %d fold cross validation on %dth fold..." % (foldsNum, i+1))
            columns.append('fold' + str(i+1))

            rawDfCopy = rawDf.copy()
            if i == foldsNum - 1:
                validSetDf = rawDf.loc[unitSize*i:, :]
            else:
                validSetDf = rawDf.loc[unitSize*i:unitSize*(i+1), :]

            trainSetDf = rawDfCopy.drop(validSetDf.index)
            validSetDf.reset_index(drop=True)
            trainSetDf.reset_index(drop=True)
            idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)

            if modelType == 'lr':
                lr = LogisticRegression()

                learningRate = paramsDict['learningRate'] if 'learningRate' in paramsDict else 0.01
                epoch = paramsDict['epoch'] if 'epoch' in paramsDict else 23500
                lam = paramsDict['lambda'] if 'lambda' in paramsDict else 0.01

                lr.fit(featuresTrain, labelsTrain, learningRate=learningRate, epoch=epoch, lam=lam)
                labelsPredictedTrain = lr.predict(featuresTrain)
                labelsPredictedValid = lr.predict(featuresValid)

                eval = Evaluation()
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

                learningRate = paramsDict['learningRate'] if 'learningRate' in paramsDict else 0.01
                epoch = paramsDict['epoch'] if 'epoch' in paramsDict else 23500
                lam = paramsDict['lambda'] if 'lambda' in paramsDict else 0.01
                batchSize = paramsDict['batchSize'] if 'batchSize' in paramsDict else 1
                hiddenLayerSize  = paramsDict['hiddenLayerSize'] if 'hiddenLayerSize' in paramsDict else 2

                nn.fit(featuresTrain, labelsTrain, learningRate=learningRate, epoch=epoch, batchSize=batchSize, lam=lam, hiddenLayerSize=hiddenLayerSize)
                labelsPredictedTrain = nn.predict(featuresTrain)
                labelsPredictedValid = nn.predict(featuresValid)

                eval = Evaluation()
                accuracyTrain = eval.computeAccuracy(labelsPredictedTrain, labelsTrain)
                accuracyValid = eval.computeAccuracy(labelsPredictedValid, labelsValid)
                macroF1Train = f1_score(labelsTrain, labelsPredictedTrain, average='macro')  # Check with sklearn
                macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn

                accuraciesTrain.append(accuracyTrain)
                accuraciesValid.append(accuracyValid)
                f1sTrain.append(macroF1Train)
                f1sValid.append(macroF1Valid)

        accuraciesTrain.append(sum(accuraciesTrain) / foldsNum)
        accuraciesValid.append(sum(accuraciesValid) / foldsNum)
        f1sTrain.append(sum(f1sTrain) / foldsNum)
        f1sValid.append(sum(f1sValid) / foldsNum)
        columns.append('Average')
        result = pd.DataFrame(np.array([accuraciesTrain, accuraciesValid, f1sTrain, f1sValid]), columns=columns, index=index)
        print(result)


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
    lr.fit(featuresTrain, labelsTrain, learningRate=0.2, epoch=50000, lam=0)
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

    # print(labelsPredictedValid)
    # print(labelsValid)


def tuneNeuralNetwork(featuresTrain, featuresValid, labelsTrain, labelsValid):
    # Fit model and predict
    nn = NeuralNetwork()
    nn.fit(featuresTrain, labelsTrain, learningRate=0.1, epoch=50000, batchSize=1, lam=0, hiddenLayerSize=10)
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

    # print(labelsPredictedValid)
    # print(labelsValid)


def plotLearningCurve(featuresTrain, featuresValid, labelsTrain, labelsValid):
    epochs = [100, 500, 1000, 5000, 10000, 20000, 50000]
    f1s = []

    # Plot learning curve for logistic regression
    for epoch in epochs:
        lr = LogisticRegression()
        lr.fit(featuresTrain, labelsTrain, learningRate=0.2, epoch=epoch, lam=0)
        labelsPredictedTrain = lr.predict(featuresTrain)
        labelsPredictedValid = lr.predict(featuresValid)

        eval = Evaluation()
        macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn
        f1s.append(macroF1Valid)

    fig, ax = plt.subplots()
    ax.plot(epochs, f1s, label="Validation Macro F1", marker='o', markersize=2)
    ax.set(xlabel='Epoch', ylabel='F1 Score',
           title='LR F1 Score against Epoch')
    ax.set_ylim(0, max(f1s)*1.1)
    ax.set_xlim(0, max(epochs)*1.1)
    ax.grid()
    ax.legend()
    fig.savefig("LR.png")

    # Plot learning curve for neural network
    f1s = []
    for epoch in epochs:
        nn = NeuralNetwork()
        nn.fit(featuresTrain, labelsTrain, learningRate=0.1, epoch=epoch, batchSize=1, lam=0, hiddenLayerSize=10)
        labelsPredictedTrain = nn.predict(featuresTrain)
        labelsPredictedValid = nn.predict(featuresValid)

        eval = Evaluation()
        macroF1Valid = f1_score(labelsValid, labelsPredictedValid, average='macro')  # Check with sklearn
        f1s.append(macroF1Valid)

    # Plot learning curve for neural network
    fig, ax = plt.subplots()
    ax.plot(epochs, f1s, label="Validation Macro F1", marker='o', markersize=2)
    ax.set(xlabel='Epoch', ylabel='F1 Score',
           title='NN F1 Score against Epoch')
    ax.set_ylim(0, max(f1s) * 1.1)
    ax.set_xlim(0, max(epochs) * 1.1)
    ax.grid()
    ax.legend()
    fig.savefig("NN.png")



if __name__ == '__main__':
    start = time.time()

    rawDf = readDataFrame(sys.argv[1])

    # Get distribution of labels
    computeDistribution(rawDf)

    # Split data
    trainSetDf, validSetDf = sample(rawDf, random_state=1)

    # Preprocess data
    idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)

    # Baseline
    runBaseline(featuresTrain, featuresValid, labelsTrain, labelsValid)

    # Logistic regression
    # tuneLogisticRegression(featuresTrain, featuresValid, labelsTrain, labelsValid)
    # paramsDict = {'learningRate': 0.2, 'epoch': 50000, 'lambda': 0}
    # cv = CrossValidation(rawDf, foldsNum=10, modelType='lr', paramsDict=paramsDict)
    # cv.run()

    # Neural Network
    # tuneNeuralNetwork(featuresTrain, featuresValid, labelsTrain, labelsValid)
    paramsDict = {'learningRate': 0.1, 'epoch': 50000, 'lambda': 0, 'hiddenLayerSize': 10, 'batchSize': 1}
    cv = CrossValidation(rawDf, foldsNum=10, modelType='nn', paramsDict=paramsDict)
    cv.run()

    # Plot learning curve
    # plotLearningCurve(featuresTrain, featuresValid, labelsTrain, labelsValid)

    end = time.time()
    computeTime(start, end)
