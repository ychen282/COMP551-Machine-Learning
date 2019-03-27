import csv
import os
import math
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import time

  
pos_path = 'train/pos/'
neg_path = 'train/neg/'
csv_path = 'naive_prediction.csv'
test_path = 'test/'
model_path_x1 = 'model_x1.json'
model_path_x0 = 'model_x0.json'

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def read_set(folder):
    data = []
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'rb') as f:
            review = f.read().decode('utf-8').replace('\n', '').strip().lower()
            data.append([review, int(file_name.split(".")[0])])
    data = sorted(data, key = lambda x: x[1], reverse=False)
    result = []
    for d, i in data:
        result.append(d)
    return result

#read train and validation

pos = read_set(pos_path)
p_train = pos[:10000]
p_validation = pos[10000:12500]

neg = read_set(neg_path)
n_train = neg[:10000]
n_validation = neg[10000:12500]

X_train = p_train + n_train
y_train = [1]*10000+[0]*10000
X_validation = p_validation + n_validation
y_validation = [1]*2500+[0]*2500

X = pos + neg

start = time.time()
common_words =[]
common_words_tuples = get_top_n_words(X_train, 2000)
for word, frequency in common_words_tuples:
    common_words.append(word)

count_vect = CountVectorizer(stop_words="english", binary=True).fit(common_words)
X_train_counts = count_vect.transform(X_train)

def Bernoulli_naive_fit(X, y): 
  global n
  global parameter_x1
  global parameter_x0
  global num_y1 
  num_y1 = 0
  m = len(X[0])
  m1_y1 = [0]*m
  m1_y0 = [0]*m
  parameter_x1 = [0]*m
  parameter_x0 = [0]*m
  
  n = len(X)
  for i in range(n):
    if (y[i] == 1):
        num_y1 += 1
    for j in range(m):
      if (y[i] == 1):
        m1_y1[j] += X[i][j]
      else:
        m1_y0[j] += X[i][j]
        
  for j in range(m):
    p_x1_y1 = (m1_y1[j]+1)/(num_y1+2)
    p_x1_y0 = (m1_y0[j]+1)/(n-num_y1+2)
    parameter_x1[j] = math.log(p_x1_y1 / p_x1_y0)
    parameter_x0[j] = math.log((1-p_x1_y1) / (1-p_x1_y0))

Bernoulli_naive_fit(X_train_counts.toarray(), y_train)      

with open(model_path_x1, 'w') as f:
        json.dump(parameter_x1, f)
with open(model_path_x0, 'w') as f:
        json.dump(parameter_x0, f)

def Bernoulli_naive_predict(X): 
  global num_y1
  global parameter_x1
  global parameter_x0
  global n
  rows = len(X)
  m = len(X[0])
 
  prediction = [math.log(num_y1/(n - num_y1))]*rows
  for i in range(rows):
    for j in range(m):
      prediction[i] += X[i][j] * parameter_x1[j] + (1 - X[i][j]) * parameter_x0[j]
    if (prediction[i] > 0):
      prediction[i] = 1
    else:
      prediction[i] = 0
                           
  return prediction
  
X_validation_counts = count_vect.transform(X_validation)
y_validation_pred = Bernoulli_naive_predict(X_validation_counts.toarray())
print(metrics.classification_report(y_validation, y_validation_pred))

def get_accuracy(y_true, y_pred):
    length = len(y_true)
    correct = 0
    for i in range (length):
         if (y_true[i] == y_pred[i]):
            correct += 1
    print("accuracy = " + str(correct/length))
get_accuracy(y_validation, y_validation_pred)

X_test = read_set(test_path)
X_test_counts = count_vect.transform(X_test)
y_pred_test = Bernoulli_naive_predict(X_test_counts.toarray())
end = time.time()
print("running time: " + str(end - start) + "s")

with open(csv_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    for i in range(0, 25001):
        if i == 0:
            writer.writerow(['Id','Category'])
        else:
            index = i-1
            pred = y_pred_test[i-1]
            writer.writerow([str(index), str(pred)])

writeFile.close()

print("Completed!")
    

