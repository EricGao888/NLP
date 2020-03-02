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

np.random.seed(seed=0)

def tfidfEncoding(trainSetDf, validSetDf):
    docs = list(trainSetDf["text"])
    wordSet = reduce(lambda x, y: set(x).union(set(y)), docs)
    wordOrder = list(wordSet)
    wordOrder.sort()

    # Compute IDF
    idfDict = dict.fromkeys(wordSet, 0)
    for word in wordSet:
        for doc in docs:
            if word in set(doc):
                idfDict[word] += 1
    docCount = len(docs)
    for word, count in idfDict.items():
        idfDict[word] = math.log(docCount / float(count))

    # TF-IDF Encoding
    trainSetDf["text"] = trainSetDf["text"].apply(lambda x: encode(x, wordSet, idfDict, wordOrder))
    validSetDf["text"] = validSetDf["text"].apply(lambda x: encode(x, wordSet, idfDict, wordOrder))


def encode(bow, wordSet, idfDict, wordOrder):
    # Compute TF
    tfDict = dict.fromkeys(wordSet, 0)
    for word in bow:
        if word in tfDict:
            tfDict[word] += 1

    bowSize = len(bow)
    for word, count in tfDict.items():
        tfDict[word] = count / float(bowSize)

    # Compute TF-IDF and map BoW to encoded vector
    tfidf = []
    for word in wordOrder:
        tfidf.append(tfDict[word] * idfDict[word])
    return tfidf
    # return 1


def text2Bow(text):
    text = text.lower()  # Transfer to lower case
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove Punctuation
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text)  # Tokenization
    try:
        stopWords = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stopWords = set(stopwords.words('english'))
    bow = [token for token in tokens if not token in stopWords]  # Remove stopwords
    stemmer = PorterStemmer()
    bow = [stemmer.stem(word) for word in bow]  # Stemming, could be replaced by lemmatization
    return bow


def preprocess(trainSetDf, validSetDf):
    # Shuffle data
    trainSetDf = trainSetDf.sample(frac=1, random_state=1).reset_index(drop=True)
    validSetDf = validSetDf.sample(frac=1, random_state=1).reset_index(drop=True)

    # Preprocess text, value representation, test set is excluded from corpus to maintain untouched
    trainSetDf['text'] = trainSetDf['text'].apply(lambda x: text2Bow(x))  # Transfer text into BoW
    validSetDf['text'] = validSetDf['text'].apply(lambda x: text2Bow(x))
    tfidfEncoding(trainSetDf, validSetDf)  # TF-IDF encoding

    # Value Representation for issue field and author field
    # One hot encode issue and author
    # Training data and validation / test data must be one-hot-encoded together to maintain consistency
    trainCount = trainSetDf.shape[0]
    issueDf = pd.get_dummies(pd.concat([trainSetDf['issue'].to_frame(), validSetDf['issue']])['issue'].to_frame()).reset_index(drop=True)
    authorDf = pd.get_dummies(pd.concat([trainSetDf['author'].to_frame(), validSetDf['author'].to_frame()])['author']).reset_index(drop=True)

    # Transfer dataframe into numpy array for better performance
    idsTrain = np.reshape(trainSetDf['tweet_id'].values, (-1, 1))
    featuresTrain = pd.DataFrame(trainSetDf['text'].tolist()).values  # Flatten list into multiple columns
    featuresTrain = np.insert(featuresTrain, [-1], issueDf.loc[:trainCount-1, :].values, axis=1)
    featuresTrain = np.insert(featuresTrain, [-1], authorDf.loc[:trainCount-1, :].values, axis=1)
    labelsTrain = np.reshape(trainSetDf['label'].values, (-1, 1))
    idsValid = np.reshape(validSetDf['tweet_id'].values, (-1, 1))
    featuresValid = pd.DataFrame(validSetDf['text'].tolist()).values  # Flatten list into multiple columns
    featuresValid = np.insert(featuresValid, [-1], issueDf.loc[trainCount:, :].values, axis=1)
    featuresValid = np.insert(featuresValid, [-1], authorDf.loc[trainCount:, :].values, axis=1)
    labelsValid = np.reshape(validSetDf['label'].values, (-1, 1))

    return idsTrain, featuresTrain, labelsTrain, idsValid, featuresValid, labelsValid

