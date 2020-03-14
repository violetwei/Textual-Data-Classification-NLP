#
# The goal is predicting 20 categories based on text.
# Inspired by the following tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups


def fetch_X_and_Y():
    newsgroup_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    # Count occurrences of words in document data
    vectorizer = CountVectorizer()
    data_as_wordCount = vectorizer.fit_transform(newsgroup_data.data)

    # Change count to frequencies
    tfidf_transformer = TfidfTransformer()
    data_as_tfidf = tfidf_transformer.fit_transform(data_as_wordCount)

    return data_as_tfidf, newsgroup_data.target


def fetch_testing_X_and_Y():
    newsgroup_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Count occurrences of words in document data
    vectorizer = CountVectorizer()
    data_as_wordCount = vectorizer.fit_transform(newsgroup_data.data)

    # Change count to frequencies
    tfidf_transformer = TfidfTransformer()
    data_as_tfidf = tfidf_transformer.fit_transform(data_as_wordCount)

    return data_as_tfidf, newsgroup_data.target


def fetch_and_split_training_validation_test_data():

    data, target = fetch_X_and_Y()

    # -------- split data into [3 (training) : 1 (validation) : 1 (testing)] -------- #
    totalInstances = target.shape[0]
    a_fifth = int(float(1 / 5) * totalInstances)
    training_endpoint = a_fifth * 3
    validation_endpoint = training_endpoint + a_fifth

    training_data = data[0:training_endpoint, :]
    training_target = target[0:training_endpoint]

    validation_data = data[training_endpoint: validation_endpoint, :]
    validation_target = target[training_endpoint:validation_endpoint]

    testing_data = data[validation_endpoint:, :]
    testing_target = target[validation_endpoint:]

    return training_data, training_target, validation_data, validation_target, testing_data, testing_target
