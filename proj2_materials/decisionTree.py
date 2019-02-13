import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import scipy
from scipy.stats import randint
from scipy.stats import uniform
import numpy as np
from sklearn import metrics

#preprocessing
  
path1 = 'train/pos/'
path2 = 'train/neg/'
path3 = 'test/'

filelist_p = os.listdir(path1)
filelist_n = os.listdir(path2)
filelist_test = os.listdir(path3)

p=[0]*len(filelist_p)
p_train = [0]*10000
p_validation = [0]*2500

n=[0]*len(filelist_n)
n_train = [0]*10000
n_validation = [0]*2500

X_test = [0]*25000

#read train and validation
i=0
for filename in filelist_p:
    f = open(path1+filename, "r")
    s = f.read()
    s = s.replace('br',' ')
    p[i]=s
    i=i+1
p_train = p[:12000]
p_validation = p[12000:12500]

i=0 
for filename in filelist_n:
    f = open(path2+filename, "r")
    s = f.read()
    s = s.replace('br',' ')
    n[i]=s
    i=i+1
n_train = n[:12000]
n_validation = n[12000:12500]
    
X_train = p_train + n_train
y_train = [1]*12000+[0]*12000
X_validation = p_validation + n_validation
y_validation = [1]*500+[0]*500

#read test
i=0 
for filename in filelist_test:
    f = open(path3+filename, "r")
    s = f.read()
    filename = filename.replace('.txt','')
    num = int(filename)
    s = s.replace('br',' ')
    X_test[num]=s
    i=i+1


#features: tf-idf and ngram

pclf = Pipeline([
    ('vect', CountVectorizer(stop_words="english")),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', LogisticRegression(solver="lbfgs",random_state=42)),
])
    
param={}

seed = 551 
random_search = RandomizedSearchCV(pclf, param_distributions=param, cv=2, verbose = 10, random_state = seed, n_iter = 6)
random_search.fit(X_train, y_train)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
report(random_search.cv_results_)
y_pred = random_search.predict(X_validation)
print(metrics.classification_report(y_validation, y_pred))

y_pred_test = random_search.predict(X_test)

#
print(X_test[5])
with open('prediction.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    for i in range(0, 25001):
        if i == 0:
            writer.writerow(['Id','Category'])
        else:
            index = i-1
            pred = y_pred_test[i-1]
            writer.writerow([str(index), str(pred)])

writeFile.close()

    
