# Issue
# 0. Do cross validation
# 1. Use sklearn to check learing accuracy against training accuracy.
# 2. Do feature engineering for validation data
# 3. Use Embedding
# 4. Do not import tune
# 5. Use stochastic gradient descent instead of batch gradient descent
# 6. Add weights for issue and party
# 7. Use lem instead of stemming
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

    rawDf = readFile(sys.argv[1])
    trainSetDf, validSetDf = sample(rawDf, random_state=1)
    idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid = preprocess(trainSetDf, validSetDf)

    end = time.time()
    computeTime(start, end)


