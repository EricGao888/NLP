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

from utility import *
from test import *
from preprocess import *
from model import *

np.random.seed(seed=0)

def crossValidation():
    pass
    # Use sklearn to check accuracy

if __name__ == '__main__':
    rawDf = readFile(sys.argv[1])
    trainSetDf, validSetDf = sample(rawDf, 1)
    idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)
    lr = LogisticRegression()
    lr.fit(featuresTrain, labelsTrain, epoch=15000)
    labelsPredicted = lr.predict(featuresValid)
    eval = Evaluation()
    accuracy = eval.computeAccuracy(labelsPredicted, labelsValid)
    print("Accuracy for Logistic Regression: %.2f" % accuracy)


    # crossValidation()
    # text2BowTest()
    # LR()
    # NN()