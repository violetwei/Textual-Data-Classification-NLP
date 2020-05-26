from pathlib import Path
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
import matplotlib.pyplot as plt


class imdb_data:

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
                'ngram_range' : imdb_data.NGRAM_RANGE,
                'dtype' : 'int32',
                'strip_accents' : 'unicode',
                'decode_error' : 'replace',
                'analyzer' : imdb_data.TOKEN_MODE,
                'min_df' : imdb_data.MIN_DOC_FREQ,
            }
            tfidf_vectorizer = TfidfVectorizer(**kwargs)
            transformed_texts = tfidf_vectorizer.fit_transform(texts)
            # Select best k features, with feature importance measured by f_classif
            selector = SelectKBest(f_classif, k=min(imdb_data.TOP_K, transformed_texts.shape[1]))
            selector.fit(transformed_texts, labels)
            transformed_texts = selector.transform(transformed_texts).astype('float32')
            return transformed_texts

        def fetch_data(dir_path):
            def load_review(reviews_path):
                files_list = list(reviews_path.iterdir())
                reviews = []
                for filename in files_list:
                    f = open(filename, 'r', encoding='utf-8')
                    reviews.append(f.read())
                return pd.DataFrame({'review':reviews})
            pos_path = dir_path+'/pos'
            neg_path = dir_path+'/neg'
            pos_reviews, neg_reviews = load_review(Path(pos_path)), load_review(Path(neg_path))
            pos_reviews['sentiment'] = 1
            neg_reviews['sentiment'] = 0
            merged = pd.concat([pos_reviews, neg_reviews])
            merged.reset_index(inplace=True)
            return merged

        #Specify the paths to data
        data_path = './data/aclImdb'
        train_path = data_path+'/train'
        test_path = data_path+'/test'

        #Fetch data
        train_data= fetch_data(train_path)
        test_data = fetch_data(test_path)

        #Encode data
        label_encoder = preprocessing.LabelEncoder()
        train_data['sentiment'] = label_encoder.fit_transform(train_data['sentiment'])
        test_data['sentiment'] = label_encoder.fit_transform(test_data['sentiment'])

        #Use more familiar variables
        training_data   = list(train_data['review'] )
        testing_data    = list(test_data['review'] )
        training_target = list( train_data['sentiment'] )
        testing_target  = list( test_data['sentiment']  )

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
    report = classification_report(y_test, predictions, target_names=['Positive','Negative'])
    print("\nClassification Report:", report)

    # confusion matrix
    confusion_matrix = get_confusion_matrix(y_test, predictions, labels=[1, 0])
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
    X_train, X_test, y_train, y_test = imdb_data.getData()
    print("y_test is of size " + str(y_test.size) + ".")
    print(y_test)
    print("Analysis of y_test array follows.")
    targets = {}
    for e in y_test:
        if e not in targets:
            targets[e] = 1
        else:
            targets[e] += 1
    c = range(2)
    amounts = []
    for k in sorted(targets.keys()):
        print("\t("+str(k)+","+str(targets[k])+").")
        amounts.append(targets[k])
    print(c)
    print(amounts)
    plt.bar(c,amounts)
    plt.title("IMDB Classifications Distribution")
    plt.xlabel("Classifications")
    plt.ylabel("Number of a classification")
    plt.show()

if __name__ == "__main__":
    aboutTestClassifications()
