import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import plot_confusion_matrix


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as ModelPipeline
import pandas as pd
from IPython.display import display

from NewsGroup_common import newsgroup_data
from NewsGroup_common import basic_model_test
from NewsGroup_common import saveDataframe
from sklearn.ensemble import RandomForestClassifier

############### Data Processing #####################

X_train, X_test, y_train, y_test = newsgroup_data.getData()

############### Decision Tree Model #####################

random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True)
predictions, accuracy, report, confusion_matrix = basic_model_test(random_forest,X_train, X_test, y_train, y_test,"NewsGroup Random Forest")

################ Cross Validation Hyperparametre Tuning ###############################

print("\nHYPERPARAMETRE TUNING")
hyperparams = { 'n_estimators': [10, 100, 1000], 'criterion': ['gini', 'entropy'] }

optimized_model = GridSearchCV(estimator=random_forest, param_grid=hyperparams,
                                n_jobs=1, cv=3, verbose=1, error_score=1)

optimized_model.fit(X_train, y_train)

print(">>>>> Optimized params")
print(optimized_model.best_params_)

cv_results = optimized_model.cv_results_
print(">>>>>> Display the top results of grid search cv")
results_dataframe = pd.DataFrame(cv_results).sort_values(by='rank_test_score')
display( results_dataframe.head() )
saveDataframe(results_dataframe,"NewsGroup Random Forest")

prediction = optimized_model.predict(X_test)
score = np.mean(prediction == y_test)
print("Using our training-dataset optimized Random Forest model on the testing dataset for evaluating")
print("score = %f" % score)

################ Graph Reporting ###############################

# plot the confusion matrix
plot_confusion_matrix(optimized_model, X_test, y_test, normalize=None, cmap=plt.cm.Blues,)
plt.show()