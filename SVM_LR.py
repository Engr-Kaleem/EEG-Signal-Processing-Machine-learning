# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:57:31 2020

@author: KaleemUllah
"""

import numpy as np                                      # for dealing with data
from scipy.signal import butter, sosfiltfilt, sosfreqz  # for filtering
import matplotlib.pyplot as plt                         # for plotting
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data = np.load('nconcat.npy')

newdata = data[1:,1:]

newdata=np.reshape(newdata, (10*10, 12,3279))

data=np.reshape(data[1:,:], (10*10, 12,3280))
labels=data[:,0,0]
X_train = newdata[:80,:,:]

y_train = labels[:80]

# X_test = np.load('./data/test_data_56_260_1_40Hz.npy')
X_test = newdata[80:,:,:]

y_test = np.reshape(labels[80:], 20)

# #data partition
#X_train = X_train_valid[20*12:,:]
#X_valid = X_train_valid[:20*12,:]



#%%
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

XC = XdawnCovariances(nfilter=5)
output = XC.fit_transform(X_train, y_train)
output = TangentSpace(metric='riemann').fit_transform(output)
print(output.shape)
#%%
outputT = XC.fit_transform(X_test, y_test)
outputT = TangentSpace(metric='riemann').fit_transform(outputT)
print(outputT.shape)

# y_train, y_test = np.array([]), np.array([])
# y_train = y

# y_test = yT

X_train = output

X_test = outputT

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#%%
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))


#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#%%
##SVM import packages
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
####SVm with linear
###C list as [1,10]
###leave 4 groups out
clSvm = SVC(kernel = 'linear') 
p = {'kernel':('linear',), 'C':[0.001,0.1,1,10]}
clSvm= GridSearchCV(clSvm, p,cv = KFold(4))
clSvm.fit(X_train, y_train )
y_hat_Svm = clSvm.predict(X_test)
print('ACC:'+ "{0:.3f}".format(accuracy_score(y_test, y_hat_Svm)))
print('AUC:'+"{0:.3f}".format(roc_auc_score(y_test, y_hat_Svm)))
##With kerneal as rbf
###C list as [1,10]
###leave 4 groups out
clSvm = SVC(kernel = 'rbf',gamma="scale") 
parameters = {'kernel':('rbf',), 'C':[0.001,0.1,1,10]}
clSvm= GridSearchCV(clSvm, parameters,cv = KFold(4))
clSvm.fit(X_train, y_train)
y_hat_Svm = clSvm.predict(X_test)
print('ACC:'+ "{0:.3f}".format(accuracy_score(y_test, y_hat_Svm)))
print('AUC:'+"{0:.3f}".format(roc_auc_score(y_test, y_hat_Svm)))
#With kerneal as sigmoid
###C list as [1,10]
###leave 4 groups out
clSvm = SVC(kernel = 'sigmoid',gamma="scale") 
parameters = {'kernel':('sigmoid',), 'C':[0.001,0.1,1,10]}
clSvm= GridSearchCV(clSvm, parameters,cv =KFold(4))
clSvm.fit(X_train, y_train)
y_hat_Svm = clSvm.predict(X_test)
print('ACC:'+ "{0:.3f}".format(accuracy_score(y_test, y_hat_Svm)))
print('AUC:'+"{0:.3f}".format(roc_auc_score(y_test, y_hat_Svm)))
#%%  Random forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 1000,
 max_features = 3,
 max_depth = 80,
 bootstrap = True)
rf_clf.fit(X_train,y_train)
y_pred_r = rf_clf.predict(X_test)

print('ACC of Randomforest classifier on test set:'+ "{0:.3f}".format(accuracy_score(y_test, y_pred_r)))
print('AUC of Randomforest classifier on test set:'+"{0:.3f}".format(roc_auc_score(y_test, y_pred_r)))

#%%
r_roc_auc = roc_auc_score(y_test, y_pred_r)
fpr, tpr, thresholds = roc_curve(y_test, rf_clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='RandomForest (area = %0.2f)' % r_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of RandomForest (4-fold CV)')
plt.legend(loc="lower right")
plt.savefig('Randomforest_ROC')
plt.show()
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
clf_tree = DecisionTreeClassifier()
parameters = {
    'min_samples_split' : range(10,500,20),
    'max_depth': range(1,20,2)
}
dt = GridSearchCV(clf_tree, param_grid = parameters, cv = 4)
dt.fit(X_train, y_train)
dt.best_params_
dtc = DecisionTreeClassifier(max_depth = 1, min_samples_split = 10)
dtc.fit(X_train,y_train)
y_pred_d = dtc.predict(X_test)
print('ACC of Decision Tree classifier on test set:'+ "{0:.3f}".format(accuracy_score(y_test, y_pred_d)))
print('AUC of Decision Tree classifier on test set:'+"{0:.3f}".format(roc_auc_score(y_test, y_pred_d)))
#%%
dt_roc_auc = roc_auc_score(y_test, y_pred_d)
fpr, tpr, thresholds = roc_curve(y_test, dtc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Decision Tree (4-fold CV)')
plt.legend(loc="lower right")
plt.savefig('DT_ROC')
plt.show()
#%%
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
y_pred_dt = dtc.predict(X_test)
print('ACC of Decision Tree classifier on test set:'+ "{0:.3f}".format(accuracy_score(y_test, y_pred_dt)))
print('AUC of Decision Tree classifier on test set:'+"{0:.3f}".format(roc_auc_score(y_test, y_pred_dt)))
#%%
log = logreg.predict_proba(X_test)[:,1]
np.save('log_labels.npy', log)

rf_ = rf_clf.predict_proba(X_test)[:,1]
np.save('rf_labels.npy', rf_)

aa=np.load('log_labels.npy')

dt = dtc.predict_proba(X_test)[:,1]
np.save('dt_labels.npy', dt)
bb=np.load('dt_labels.npy')