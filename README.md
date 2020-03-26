# Textual-Data-Classification-NLP

Goal: develop models to classify textual data, input is text documents, and output is categorical variable

#Multi-class classification problem and text classification

## Datasets
- 20 news group dataset. Use the default train subset (subset=‘train’, and remove=([‘headers’, ‘footers’, ‘quotes’]) in sklearn.datasets) to train the models and report the final performance on the test subset. Note: need to start with the text data and convert text to feature vectors. Please refer to https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html for a tutorial on the steps needed for this.

- IMDB Reviews: http://ai.stanford.edu/~amaas/data/sentiment/ Here, you need to use only reviews in the train folder for training and report the performance from the test folder. You need to work with the text documents to build your own features and ignore the pre-formatted feature files.

## Models
Apply and compare the performance of following models:
- Logistic regression: sklearn.linear model.LogisticRegression 
- Decision trees: sklearn.tree.DecisionTreeClassifier
- Support vector machines: sklearn.svm.LinearSVC
- Ada boost: sklearn.ensemble.AdaBoostClassifier
- Random forest: sklearn.ensemble.RandomForestClassifier
- Naive Bayes: sklearn.naive_bayes.MultinomialNB

Use any Python libraries to extract features and preprocess the data, and to tune the hyper-parameters

## Validation
Develop a model validation pipeline (e.g., using k-fold cross validation or a held-out validation set) and study the effect of different hyperparamters or design choices. In a single table, compare and report the performance of the above mentioned models (with their best hyperparameters), and mark the winner for each dataset and overall.
