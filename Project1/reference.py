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
        text = test_tweet_id2text[tweet_id]

        # Predict the label
        label = randrange(1, 18)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id] = label

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
        text = test_tweet_id2text[tweet_id]

        # Predict the label
        label = randrange(1, 18)

        # Store it in the dictionary
        test_tweet_id2label[tweet_id] = label

    # Save predicted labels in 'test_lr.csv'
    SaveFile(test_tweet_id2text, test_tweet_id2issue, test_tweet_id2author_label, test_tweet_id2label, 'test_nn.csv')

