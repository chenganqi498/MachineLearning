import csv
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score

data = genfromtxt('../forest_train.csv', delimiter=',')
data_val = genfromtxt('../forest_validation.csv', delimiter=',')

tr_sub = range(0, 1000)
val_sub = range(0, 1000)

X_cont = preprocessing.scale(data[tr_sub,0:10])
X_bin = data[tr_sub, 10:50]
X_new = np.hstack([X_cont, X_bin])
y = data[tr_sub, 50]

X_val_cont = preprocessing.scale(data_val[val_sub,0:10]) 
X_val_bin = data_val[val_sub, 10:50]
X_val_new = np.hstack([X_val_cont, X_val_bin])
y_val = data_val[val_sub, 50]

clf_ex = ExtraTreesClassifier(n_estimators=80, max_features = 10, \
                              max_depth=None, min_samples_split=1, \
                              random_state=0)
                              
# 3-fold cross validation score                              
scores = cross_val_score(clf_ex, X_new, y).mean()

fit_ex = clf_ex.fit(X_new, y)
val_sc = fit_ex.score(X_val_new, y_val)

# Put train and validation set together 
X_tol = np.vstack((X_new, X_val_new))
y_tol = np.hstack((y, y_val))
fit_tol = clf_ex.fit(X_tol, y_tol)

# Test on the test set!
data_test = genfromtxt('../forest_test.csv', delimiter=',')
X_test_scaled = preprocessing.scale(data_test[:,0:10]) 
X_test_new = np.hstack([X_test_scaled, data_test[:,10:50]])
ypred_test = fit_tol.predict(X_test_new)

# Output
ID = np.array(range(1, len(ypred_test)+1))
Out = np.column_stack((ID, ypred_test)).astype(int)
b = open('prediction.csv', 'w')
a = csv.writer(b)
a.writerows(Out)
b.close()
