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


def readFile(inputFilePath):
    rawDf = pd.read_csv(inputFilePath, header=[0])
    return rawDf

def saveFile(outputFilePath, originalDf, id2LabelDict):
    # Use zip ---> list of tuples ---> dataframe to pack data for output
    pass

def sample(rawDf, random_state):
    rawDfCopy = rawDf.copy()
    trainSetDf = rawDf.sample(frac=0.8, random_state=random_state).reset_index(drop=True)
    validSetDf = rawDfCopy.drop(trainSetDf.index).reset_index(drop=True)
    return trainSetDf, validSetDf

