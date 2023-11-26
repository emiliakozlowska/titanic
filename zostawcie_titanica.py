import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV #Recursive feature elimination with cross-validation to select the number of features.
from sklearn.model_selection import GridSearchCV #GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets.
from sklearn.ensemble import RandomForestClassifier #A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
from sklearn.pipeline import Pipeline #Implements utilities to build a composite estimator, as a chain of transforms and estimators.
from sklearn.metrics import classification_report #Build a text report showing the main classification metrics.
from sklearn import metrics #Module includes score functions, performance metrics and pairwise metrics and distance computations.
from sklearn.metrics import roc_auc_score #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

df = pd.read_csv('dane.csv', sep=';')
df = df.dropna(how='any') #.dropna() - Remove missing values.; 0/index - Drop rows which contain missing values. ‘any’:if any NA values are present, drop that row or column.
df.head() #This function returns the first n (default n=5) rows for the object based on position


y = df['Survived'] #dependent variable
X = df[['Age', 'Pclass', 'Fare', 'Sex']] #independent variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#test_size - If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
#random_state - Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

#Creating the RandomForestClassifier model nr1, the classifier used for feature selection.
RFC1 = RandomForestClassifier(n_estimators=30, random_state=42, class_weight="balanced")
#The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
RFECV = RFECV(estimator=RFC1, scoring = 'roc_auc') 
#Recursive feature elimination with cross-validation to select the number of features.

#Creating the RandomForestClassifier model nr2, the classifier used for feature selection.
RFC2 = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced") 
#Creating a grid for the RFC2 model
Grid_RFC2 = GridSearchCV(RFC2, param_grid={'max_depth':[2,3]}, scoring = 'roc_auc')
#param_grid - Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of parameter settings.

#Pipeline of transforms with a final estimator.
pipeline = Pipeline([('RFECV',RFECV), ('Grid_RFC2',Grid_RFC2)])

pipeline.fit(X_train, y_train) #.fit() Fit all the transformers one after the other and transform the data. Finally, fit the transformed data using the final estimator.
#Fit the pipeline on the training data and use it to predict topics for X_test. 
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:,1]

#Compute confusion matrix to evaluate the accuracy of a classification.
co_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion_matrix\n',co_matrix)

print('Classification report\n',classification_report(y_test, y_pred)) #Build a text report showing the main classification metrics for each class. 

print("Accuracy: ",np.round(metrics.accuracy_score(y_test, y_pred), 2)) 
print("Precision: ",np.round(metrics.precision_score(y_test, y_pred), 2))
print("Recall: ",np.round(metrics.recall_score(y_test, y_pred), 2))
print("F1 score: ",np.round(metrics.f1_score(y_test, y_pred), 2))

#Parameter evaluation
print('Best params:', Grid_RFC2.best_params_)
print('Best score:',np.round(Grid_RFC2.best_score_,2))
print('Best estimator:',Grid_RFC2.best_estimator_)

#roc_curve - Compute Receiver operating characteristic
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba) 
#roc_auc_score - Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
auc = metrics.roc_auc_score(y_test, y_pred_proba) 

#ROC chart
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.2f)' % auc)
plt.xlabel('False Positive Rate',color='grey', fontsize = 14)
plt.ylabel('True Positive Rate',color='grey', fontsize = 14)
plt.title('ROC - receiver operating characteristic')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1],'r--')
plt.show()

print('Roc auc score:', round(roc_auc_score(y_test, y_pred),2))