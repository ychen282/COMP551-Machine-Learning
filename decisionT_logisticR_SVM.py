import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.decomposition import PCA
import time


#==============================================================================
# Data Preprocessing
#==============================================================================
  
path1 = 'train/pos/'
path2 = 'train/neg/'
path3 = 'test/'

filelist_p = os.listdir(path1)
filelist_n = os.listdir(path2)
filelist_test = os.listdir(path3)

p=[0]*len(filelist_p)

#=========================================
#Train and Validation split
#========================================
#number of training data: nt
#number of validation data: nv
#when generating prediction csv file, uncomment the Validation part and change nt to 12500
nt = 10000
nv = 12500-nt
p_train = [0]*nt
p_validation = [0]*nv


n=[0]*len(filelist_n)
n_train = [0]*nt
n_validation = [0]*nv

X_test = [0]*25000

#read train and validation
i=0
for filename in filelist_p:
    f = open(path1+filename, "r")
    s = f.read()
    s = s.replace('br',' ')
    s = s.replace(' the ',' ')
    s = s.replace(' is ',' ')
    s = s.replace(' are ',' ')
  
    p[i]=s
    i=i+1
p_train = p[:nt]
p_validation = p[nt:12500]

i=0 
for filename in filelist_n:
    f = open(path2+filename, "r")
    s = f.read()
    s = s.replace('br',' ')
    s = s.replace(' the ',' ')
    s = s.replace(' is ',' ')
    s = s.replace(' are ',' ')
    n[i]=s
    i=i+1
n_train = n[:nt]
n_validation = n[nt:12500]
    
X_train = p_train + n_train
y_train = [1]*nt+[0]*nt
X_validation = p_validation + n_validation
y_validation = [1]*nv+[0]*nv

#read test
i=0 
for filename in filelist_test:
    f = open(path3+filename, "r")
    s = f.read()
    filename = filename.replace('.txt','')
    num = int(filename)
    s = s.replace('br',' ')
    s = s.replace(' the ',' ')
    s = s.replace(' is ',' ')
    s = s.replace(' are ',' ')
    X_test[num]=s
    i=i+1

#==============================================================================
# Build model
#==============================================================================


start = time.time()

pclf = Pipeline([
   ('vect', CountVectorizer(strip_accents='ascii')),
#=============================================================
# Uncoment line the following two lines to include tfidf
#=============================================================
 #  ('tfidf', TfidfTransformer()),
   ('norm', Normalizer()),
#=============================================================
# Uncoment the following lines to test DecisionTree
#=============================================================
 # ('clf', tree.DecisionTreeClassifier())
 # ('clf', tree.DecisionTreeClassifier(max_depth=500))
 # ('clf', tree.DecisionTreeClassifier(max_depth=500))
#=============================================================
# Uncoment the following line to test LogisticRegression
#=============================================================
#   ('clf', LogisticRegression(solver="lbfgs",random_state=42)),
#=============================================================
# Uncoment the follwing line to test Linear SVC
#=============================================================
  ('clf', LinearSVC(random_state=0, tol=1e-5)),
  ])

#======================================================================
# params list, uncomment to find the best parameter set
# it is now commented to save run time, but we picked out the result of
# the best parameters set to be used
#======================================================================  
#params = {"vect__ngram_range": [(1,1), (1,2), (2,2)],
#          "vect__binary": [True],
#          "vect__max_df": [0.6, 0.8, 1.0]
#         }

params = {"vect__ngram_range": [(1,2)],
          "vect__binary": [True],
          "vect__max_df": [0.8]
          }

grid_search = GridSearchCV(pclf, param_grid = params, cv=5)
grid_search.fit(X_train, y_train)

#==============================================================================
# Validation
#==============================================================================
y_pred = grid_search.predict(X_validation)
print(metrics.classification_report(y_validation, y_pred))
print("Best parameter: ", grid_search.best_params_)

def accuracy(y_true, y_pred):
    length = len(y_true)
    correct = 0
    for i in range (length):
         if (y_true[i] == y_pred[i]):
            correct += 1
    print("accuracy = " + str(correct/length))
    
accuracy(y_validation, y_pred)

end = time.time()
print("running time: " + str(end - start) + "s")

#==============================================================================
# Predict test set and write csv
#==============================================================================

y_pred_test = grid_search.predict(X_test)

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

    
