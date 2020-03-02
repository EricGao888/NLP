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

from main import *

def text2BowTest():
    str = "This is a ** %% strange *** st((ring to test text2Bow function!"
    print(str)
    print(text2Bow(str))

if __name__ == '__main__':
    pass