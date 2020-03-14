from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt



class newsgroup_data:

    # Prepare the data
    # select the top 20000 features from the vector of tokens
    NGRAM_RANGE = (1, 2)
    TOP_K = 20000
    TOKEN_MODE = 'word'
    MIN_DOC_FREQ = 2

    @staticmethod
    def getData():

        def ngram_vectorize(texts, labels):
            kwargs = {
                'ngram_range' : newsgroup_data.NGRAM_RANGE,
                'dtype' : 'int32',
                'strip_accents' : 'unicode',
                'decode_error' : 'replace',
                'analyzer' : newsgroup_data.TOKEN_MODE,
                'min_df' : newsgroup_data.MIN_DOC_FREQ,
            }
            tfidf_vectorizer = TfidfVectorizer(**kwargs)
            transformed_texts = tfidf_vectorizer.fit_transform(texts)
            # Select best k features, with feature importance measured by f_classif
            selector = SelectKBest(f_classif, k=min(newsgroup_data.TOP_K, transformed_texts.shape[1]))
            selector.fit(transformed_texts, labels)
            transformed_texts = selector.transform(transformed_texts).astype('float32')
            return transformed_texts

        # Get the training and testing datasets
        training_set = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        testing_set = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
        training_data = training_set.data
        training_target = list(training_set.target)
        testing_data = testing_set.data
        testing_target = list(testing_set.target)

        # Temporarily combine the two datasets (albeit in a way that we can separate them after)
        training_length = len(training_data)
        training_data.extend(testing_data)
        training_target.extend(testing_target)
        all_data = training_data
        all_target = training_target

        # Vectorize the full dataset
        vectorized_all_data = ngram_vectorize(all_data,all_target)
        print("\nVectorized all data shape: ", vectorized_all_data.shape )

        # Reseparate the datasets
        training_data = vectorized_all_data[:training_length]
        training_target = all_target[:training_length]
        testing_data = vectorized_all_data[training_length:]
        testing_target = all_target[training_length:]
        print("\nVectorized training data shape: ",training_data.shape)
        print("\nVectorized training data shape: ",testing_data.shape)

        #Formalize the datasets
        X_train = training_data.toarray()
        y_train = np.array(training_target)
        X_test  = testing_data.toarray()
        y_test  = np.array(testing_target)

        #Return the partitions
        return X_train, X_test, y_train, y_test



from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix as get_confusion_matrix

def basic_model_test(model,X_train,X_test,y_train,y_test,name):
    
    print("model fitting started")

    # fit the model
    model.fit(X_train, y_train)

    # model prediction
    print("Starting model Prediction")
    predictions = model.predict(X_test)

    # evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("\n"+name+" Model Classification Accuracy Score:", accuracy)

    # Classification report
    target_names = []
    for i in range(20):
        target_names.append(str(i))
    report = classification_report(y_test, predictions, target_names=target_names )
    print("\nClassification Report:", report)

    # confusion matrix
    confusion_matrix = get_confusion_matrix(y_test, predictions, labels=range(20))
    print(name+" Model Confusion Matrix: \n", confusion_matrix)

    return predictions, accuracy, report, confusion_matrix

def saveDataframe(dataframe, modelname,foldername="output"):
        #Adjust filename as needed
        filename = modelname.replace(" ","_") +"_"+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #Check that the save directory exists
        outdir = "./"+foldername
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        #Save
        full_relative_path = "./" + foldername + "/" + filename + ".cvdata"
        dataframe.to_csv( full_relative_path , header=True )



def aboutTestClassifications():
    X_train, X_test, y_train, y_test = newsgroup_data.getData()
    print("y_test is of size " + str(y_test.size) + ".")
    print(y_test)
    print("Analysis of y_test array follows.")
    targets = {}
    for e in y_test:
        if e not in targets:
            targets[e] = 1
        else:
            targets[e] += 1
    classifications = range(20)
    amounts = []
    for k in sorted(targets.keys()):
        print("\t("+str(k)+","+str(targets[k])+").")
        amounts.append(targets[k])
    plt.bar(classifications,amounts)
    plt.title("20 Newsgroups Classifications Distribution")
    plt.xlabel("Classifications")
    plt.ylabel("Number of a classification")
    plt.show()

if __name__ == "__main__":
    aboutTestClassifications()