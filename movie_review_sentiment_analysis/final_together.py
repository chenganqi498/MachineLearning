import csv
import numpy as np
import scipy as sp
from random import randrange
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import KMeans, MiniBatchKMeans


#---------------------------Load and prepare data------------------------------#
data_train = load_files('train')
data_unsup = load_files('unsup')

X_train = data_train.data[0:1000]
y_train = data_train.target[0:1000]
X_test = data_train.data[1000:1500]
y_test = data_train.target[1000:1500]
X_unsup = data_unsup.data[0:5000]

Vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5)

'''
#--------------------------Train supervised data-------------------------------#

X_train_tr = Vectorizer.fit_transform(X_train)
X_test_tr = Vectorizer.transform(X_test)

clf_SVC = LinearSVC(loss='l1',dual = True, tol = 1e-3)
clf_Ridge = RidgeClassifier(tol=1e-2, solver="lsqr")
#clf_NB = MultinomialNB()

#3-fold cross_validation scores, score = (mean, std)
#cr_scores = cross_val_score(clf, X_train, y_train)
#score = [cr_scores.mean(), np.std(cr_scores)]
#duration = time() - t0

fit_SVC = clf_SVC.fit(X_train_tr, y_train)
fit_Ridge = clf_Ridge.fit(X_train_tr, y_train)
#fit_NB = clf_NB.fit(X_train_vec, y_train)

score1_SVC = fit_SVC.score(X_test_tr, y_test)
score1_Ridge = fit_Ridge.score(X_test_tr, y_test)

#y_unsup_SVC = fit_SVC.predict(X_unsup)
#y_unsup_Ridge = fit_Ridge.predict(X_unsup)
#y_unsup_NB = fit_NB.predict(X_unsup_vec)
'''
#--------------------------Cluster unsupervised data---------------------------#
'''
data_process = Pipeline([\
               ('Tfidf', Vectorizer), \
               ('lsa', TruncatedSVD(100)), \
               ('normalize', Normalizer(copy=False))
               ])
'''

'''
X_train_un = Vectorizer.transform(X_train)
X_test_un = Vectorizer.transform(X_test)

neg_loc = np.where(y_train == 0)[0]
pos_loc = np.where(y_train == 1)[0]

#init = np.vstack([X_train_un[neg_loc[randrange(0,len(neg_loc))]].mean(axis = 0), \
#                 X_train_un[pos_loc[randrange(0,len(pos_loc))]].mean(axis = 0)])

init = np.vstack([X_train_un[neg_loc].mean(axis = 0), \
                  X_train_un[pos_loc].mean(axis = 0)])
                  
km = MiniBatchKMeans(n_clusters = 2, init = init, n_init = 1,
                                   init_size=3000, batch_size=1000)
                                   
cluster = km.fit(X_unsup_un)

y_test_unsup = cluster.predict(X_test_un)
score2 = np.mean(y_test_unsup == y_test)
'''
#-----------------Combine supervised and unsupervised data---------------------#
'''
a1 = np.where(y_cluster == y_unsup_NB)[0]
a2 = np.where(y_cluster == y_unsup_Ridge)[0]
a3 = np.where(y_cluster == y_unsup_SVC)[0]
sub = list(set(a1)&set(a2)&set(a3))
y_unsup_sub = y_cluster[sub]
X_unsup_sub_vec = X_unsup_vec[sub]

X_comb_vec = sp.sparse.vstack([X_train_vec, X_unsup_sub_vec])
y_comb = np.hstack([y_train, y_unsup_sub])

fit_comb = clf_SVC.fit(X_comb_vec, y_comb)
score3 = fit_comb.score(X_test_vec, y_test)

#--------------------Predict test data and output------------------------------#

data_test = load_files('test')
X_test_vec = vectorizer.transform(data_test.data)
y_test_comb = fit_comb.predict(X_test_vec)
y_test_train = fit_Ridge.predict(X_test_vec)

# Output
Id = np.array(range(1, len(y_test_train)+1))
Out = np.column_stack((Id, y_test_train)).astype(int)
b = open('prediction_train.csv', 'w')
a = csv.writer(b)
a.writerows(Out)
b.close()
'''

