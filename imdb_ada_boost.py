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

from imdb_common import imdb_data
from imdb_common import basic_model_test
from imdb_common import saveDataframe
from sklearn.ensemble import AdaBoostClassifier

warnings.simplefilter('ignore')
sns.set(rc={'figure.figsize' : (12, 6)})
sns.set_style("darkgrid", {'axes.grid': True})

############### Data Processing #####################

X_train, X_test, y_train, y_test = imdb_data.getData()

############### Ada Boost Model #####################

ada_boost = AdaBoostClassifier(n_estimators=50,learning_rate=1.0)
predictions, accuracy, report, confusion_matrix = basic_model_test(ada_boost,X_train, X_test, y_train, y_test,"IMDB AdaBoost")

################ Cross Validation Hyperparametre Tuning ###############################

print("\nHYPERPARAMETRE TUNING")
hyperparams = {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]}

optimized_model = GridSearchCV(estimator=ada_boost, param_grid=hyperparams,
                                n_jobs=1, cv=3, verbose=1, error_score=1)

optimized_model.fit(X_train, y_train)

print(">>>>> Optimized params")
print(optimized_model.best_params_)

cv_results = optimized_model.cv_results_
print(">>>>>> Display the top results of grid search cv")
results_dataframe = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
display( results_dataframe.head() )
saveDataframe(results_dataframe,"IMDB AdaBoost")

prediction = optimized_model.predict(X_test)
score = np.mean(prediction == y_test)
print("Using our training-dataset optimized adaboost model on the testing dataset for evaluating")
print("score = %f" % score)

################ Graph Reporting ###############################

# plot the confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=False, figsize=(12, 8))
plt.show()