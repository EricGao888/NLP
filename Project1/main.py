import os
import sys
import re
import csv
import string

import pandas as pd
import numpy as np
import nltk
from random import randrange
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class LogisticRegression:
    def fit(self, features, labels):
        pass
    def predict(self, features):
        pass

class NeuralNetwork:
    def fit(self, features, labels):
        pass
    def predict(self, features):
        pass

class Evaluation:
    def computeMacroF1(self):
        pass
    def computeAccuracy(self):
        pass

def readFile(inputFilePath):
    rawDf = pd.read_csv(inputFilePath, header=[0])
    return rawDf

def sample(rawDf, random_state):
    rawDfCopy = rawDf.copy()
    trainSetDf = rawDf.sample(frac=0.8, random_state=random_state)
    validSetDf = rawDfCopy.drop(trainSetDf.index)
    return trainSetDf, validSetDf

def tfIdfEncoding(trainSetDf, validSetDf):
    return trainSetDf, validSetDf

def text2BoW(text):
    text = text.lower()  # Transfer to lower case
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove Punctuation
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(text) # Tokenization
    try:
        stopWords = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stopWords = set(stopwords.words('english'))
    bow = [token for token in tokens if not token in stopWords]  # Remove stopwords
    stemmer = PorterStemmer()
    bow = [stemmer.stem(word) for word in bow] # Stemming, could be replaced by lemmatization
    return bow

def text2BoWTest():
    str = "This is a ** %% strange *** st((ring to test text2BoW function!"
    print(str)
    print(text2BoW(str))

def preprocess(trainSetDf, validSetDf):
    trainSetDf["text"] = trainSetDf["text"].apply(lambda x: text2BoW(x))
    validSetDf["text"] = validSetDf["text"].apply(lambda x: text2BoW(x))
    print(trainSetDf["text"].head(3))

def saveFile(dataframe, outputFilePath):
    return 0

def logisticRegression():

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Logistic Regression classifier here
    '''

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')

    # Predict test data by learned model

    '''
    Replace the following random predictor by your prediction function.
    '''

    for tweet_id in test_tweet_id2text:
        # Get the text
        text=test_tweet_id2text[tweet_id]

        # Predict the label
        label=randrange(1, 18)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id]=label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_lr.csv')


def neuralNetwork():

    # Read training data
    train_tweet_id2text, train_tweet_id2issue, train_tweet_id2author_label, train_tweet_id2label = ReadFile('train.csv')

    '''
    Implement your Neural Network classifier here
    '''

    # Read test data
    test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label = ReadFile('test.csv')

    # Predict test data by learned model
    # Replace the following random predictor by your prediction function
    
    for tweet_id in test_tweet_id2text:
        # Get the text
        text=test_tweet_id2text[tweet_id]

        # Predict the label
        label=randrange(1, 18)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id]=label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_nn.csv')

if __name__ == '__main__':
    rawDf = readFile(sys.argv[1])
    trainSetDf, validSetDf = sample(rawDf, 0)
    preprocess(trainSetDf, validSetDf)
    text2BoWTest()
    # LR()
    # NN()