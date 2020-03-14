import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, StratifiedKFold
import scikitplot as skplt

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as ModelPipeline
import pandas as pd
from IPython.display import display

from NewsGroup_common import newsgroup_data
from NewsGroup_common import basic_model_test
from NewsGroup_common import saveDataframe
from sklearn.naive_bayes import MultinomialNB

############### Data Processing #####################

X_train, X_test, y_train, y_test = newsgroup_data.getData()

############### Decision Tree Model #####################

naive_bayes = MultinomialNB(alpha=1.0)
predictions, accuracy, report, confusion_matrix = basic_model_test(naive_bayes,X_train, X_test, y_train, y_test,"NewsGroup Multinomial Naive Bayes")

################ Cross Validation Hyperparametre Tuning ###############################

print("\nHYPERPARAMETRE TUNING")
hyperparams = {'alpha': (1e0,1e-2, 1e-4, 1e-10)}

optimized_model = GridSearchCV(estimator=naive_bayes, param_grid=hyperparams,
                                n_jobs=1, cv=3, verbose=1, error_score=1)

optimized_model.fit(X_train, y_train)

print(">>>>> Optimized params")
print(optimized_model.best_params_)

cv_results = optimized_model.cv_results_
print(">>>>>> Display the top results of grid search cv")
results_dataframe = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
display( results_dataframe.head() )
saveDataframe(results_dataframe,"NewsGroup Multinomial Naive Bayes")

prediction = optimized_model.predict(X_test)
score = np.mean(prediction == y_test)
print("Using our training-dataset optimized multinomial naive bayes model on the testing dataset for evaluating")
print("score = %f" % score)

################ Graph Reporting ###############################

# plot the confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=False, figsize=(12, 8))
plt.show()
