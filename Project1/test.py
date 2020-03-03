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

from main import *
from utility import *
from model import *


def text2BowTest():
    str = "This is a ** %% strange *** st((ring to test text2Bow function!"
    print(str)
    print(text2Bow(str))


def computeTimeTest():
    start = time.time()
    time.sleep(2)
    end = time.time()
    computeTime(start, end)

if __name__ == '__main__':
    computeTimeTest()