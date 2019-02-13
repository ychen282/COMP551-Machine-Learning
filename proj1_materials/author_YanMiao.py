import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import matplotlib.pyplot as plt # visualization
import numpy as np
from numpy import *
from numpy import linalg as LA # calculate norm
#import string # process text comment
import pandas as pd
from sklearn import preprocessing # used for feature scaling
import time # time different methods
from collections import Counter # select most frequent elements in a list
import tabulate 
import seaborn as sns # visulization
from copy import deepcopy # used to copy the created list of dictionary


total_start = time.time()

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

# Example:
data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))

    
def text_modification(data):
    for x in data:
        x['text'] = x['text'].lower()
#        for y in string.punctuation:
#            x['text'] = x['text'].replace(y, '')
        x['text'] = x['text'].split(' ')
    return data
data = text_modification(data)


# Top frequent words 
# "data" is the input list of dictionaries, and "num" is the number of top frequent words needed
def top_frequent_words(data, num):
    words = []
    for x in data:
        words = words + x['text']
        
    # Find 160 most frequent words
    Count = Counter(words) 
    most_occur = Count.most_common(num)
    
    # Get rid of the tuple in the list most_occur1 and most_occur2
    words = []
    for x in most_occur:
        words.append(x[0])

    return words
    
wordList = top_frequent_words(data, 60) # 160 word will lead to overfitting and lower speed

# Training, Validation, Test sets split
training, validation, test = np.split(data, [int((5/6)*len(data)), int((11/12)*len(data))])
#training, validation, test = np.split(data, [int((1/2)*len(data)), int((3/4)*len(data))])

# Create the "word count features"
def word_list(data, wordList):
    word_matrix = [] # a matrix shows the frequency of the top words occurring in comments
    for x in wordList:    
        temp = []
        for y in data:
            num = y['text'].count(x)
            temp.append(num)
        word_matrix.append(temp)

    return word_matrix  

# Find the correlation between word frequency and popularity score in each comment
def help_sort(elem):
    return elem[1]

def dot(K, L):
   if len(K) != len(L):
       print("Dimensions not equal")
       return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def word_corr(wordList, X_pop, data):

    word_score_corr = [] # list of "["word1", corr_value]"
    i = 0
    for word in wordList:
        list1 = [1 if word in data[i]['text'] else -1 for i in range(0,len(X_pop))]
        list2 = X_pop - sum(X_pop)/len(X_pop)
        corr = dot(list1, list2) / len(X_pop)
        word_score_corr.append((wordList[i], corr))
        i = i+1    
    
    # sort the list according to correlation coefficient of each word
    word_score_corr = sorted(word_score_corr, key=help_sort)
    
    # Pick out 20 words, 10 with the highest positive correlation coefficient and 10 with the lowest
    top_corr_words = []
    
    for y in word_score_corr[-5:]:
        top_corr_words.append(y)
        
    for x in word_score_corr[0:4]:
        top_corr_words.append(x)
        
    # Calculate the average of sum of correlation score in each comment
    sum_list = []
    for x in data:
        score = 0
        for words in x['text']:
            for elem in top_corr_words:
                if elem[0]==words:
                    score += elem[1]
                else:
                    score += 0
#        score = score/len(x['text'])
        sum_list.append(score)
        
    return sum_list
      
def feature_score(data):
    # Plot the graph of any feature and popularity score
    length = [] # Any of the feature
    scores = []
    
    for x in data:
        length.append(np.power(len(x['text']),1.1))
        scores.append(x['popularity_score'])
    
    fig = plt.figure()
    plt.plot(length, scores, 'k.')
    fig.suptitle("Comment length & popularity score", fontsize=16)
    plt.xlabel('length', fontsize=14)
    plt.ylabel('popularity score', fontsize=14)
    plt.show()

  
# Feature and label extraction 
def information_extraction(data):
    text = []
    is_root = []
    controversiality = []
    children = []
    log_children = []
    popularity_score = []
    
    for x in data:
        text.append(x['text'])
        if x['is_root']==False:
            is_root.append(0)
        else:
            is_root.append(1)
        controversiality.append(x['controversiality'])
        children.append(x['children'])
        log_children.append(np.log10(x['children']+1))
        
        popularity_score.append(x['popularity_score'])
        
    return text, is_root, controversiality, children, log_children, popularity_score

# Create dummy variable list
def onelist(n):
    listofones = [1] * n
    return listofones

# Create a new feature: number of words in each comment
def num_words(X_text):
    X_numwords = []
    
    for x in X_text:
        X_numwords.append(len(x))
        
    return X_numwords

# Create a new feature: the average length of words in each comment
def word_avglen(X_text):
    X_avglen = []
    for wordlists in X_text:
        total_length = 0
        for words in wordlists:
            total_length += len(words)
        avglen = total_length
        X_avglen.append(avglen)
        
    return X_avglen

# Create a new feature: find #words that belong to the most frequent 160 words in each text
def intersection(list1, list2): 
    list3 = [value for value in list1 if value in list2]
    card = len(list3)
    return list3, card

# Count number of most frequent words in the comment text
def num_frequent_words(X_text):
    X_intersect_card = []
    for x in X_text:
        intersect_list, intersect_card = intersection(x,wordList)
        X_intersect_card.append(intersect_card)
    return X_intersect_card

    
## Using least-squares

def error_function(W_est, X, y):
    # Calculate MSE
    totalError = np.dot((y-np.dot(X, W_est)).T, y-np.dot(X, W_est))
    return totalError[0][0]/len(y)

def least_squares_runner(X, y):
    XTX = X.T.dot(X)
    
    if np.linalg.det(XTX) == 0.0:
        # If the matrix is singular, this shows that some of the features 
        # are not linearly independent
        print("Singular matrix error")
        return 
    
    XTX_inv = pd.DataFrame(np.linalg.inv(XTX.values), XTX.columns, XTX.index)
    
    # get estimation of weight matrix
    W_est = XTX_inv.dot(X.T.dot(y))
    
    err = error_function(W_est, X, y)
    return err, W_est


## Using gradient descent

def gradient_descent_setup(X, y, eta_0, beta, epsilon):
    # Initialize W with zeros
    W_0 = np.random.random((X.shape[1],1))
    #W_0 = np.zeros(shape=(X.shape[1],1))
    W_est, err_record = gradient_descent_runner(X, y, W_0, beta, eta_0, epsilon)
    err = error_function(W_est, X, y)
    return W_est, err_record, err
    
    
def gradient_descent_runner(X, y, W_0, beta, eta_0, epsilon):
    W = [W_0, 1+W_0]

    i = 1
    err_record = []
    
    # For 60 words
    const = 1.3 # constant used in weight matrix normalization
    
    # For 160 words
    #const = 0.86
    
    while LA.norm(W[1]-W[0]) > epsilon:
        if i != 1:
            W[0] = W[1]
        alpha = eta_0 / (1 + beta*i)
        W[1] = W[0] - 2 * alpha * (np.dot(np.dot(X.T,X), W[0]) - np.dot(X.T, y))
        # gradient clipping
        W[1] = const * W[1] / np.linalg.norm(W[1]) # normalize W[1] and time a constant
        i+=1
        
        if i % 3 == 0:
            err_record.append(error_function(W[1], X, y))
        
        if i > 100000:
            break   
    
    return W[1], err_record


def data_preprocessing(data):
    # Create training sets
    X_text, X_root, X_contro, X_chil, X_logchil, X_pop = information_extraction(data)
    
    #feature_score(data)
    
    # The reason I partition features into severl lists is to make it convenient when
    # we choose to discard one or more features
    ones = onelist(len(X_pop))
    X_numwords_0 = num_words(X_text)

    X_avglen_0 = word_avglen(X_text)

    #X_intersect_card = num_frequent_words(X_text)
    
    word_matrix = word_list(data, wordList)
    sum_list = word_corr(wordList, X_pop, data)
    
    # Feature scaling (only for the new features)
    X_numwords = preprocessing.scale(X_numwords_0)
    X_numwords_log = preprocessing.scale(np.log([x+1 for x in X_numwords_0]))
    
    #X_sumlist = preprocessing.scale(sum_list)
    
    X_avglen = preprocessing.scale(X_avglen_0)
    X_avglen_log = preprocessing.scale(np.log([x+1 for x in X_avglen_0]))

    
    #X_intersect_card = preprocessing.scale(np.log([x+1 for x in X_intersect_card]))

    # Merge feature lists into dataframe
    X_train = pd.DataFrame(
        {#'text': X_text,
         'is_root': X_root,
         'controversiality': X_contro,
         'children': X_chil,
         'log_children': X_logchil,
         #'num_words': X_numwords, # new feature
         
         #'num_words_log': X_numwords_log,
         
         #'sum_list': X_sumlist,
         'sum_list': np.log([x+1 for x in sum_list]),

# =============================================================================
#          'sum_list_3': np.power(sum_list,3),
#          'words_avglen': X_avglen,
#          
          'words_avglen_log': X_avglen_log,
#          
#          'intersection_cardinality': X_intersect_card, # new feature
# =============================================================================
         'dummy var': ones
        })
    

    i = 0
    for x in word_matrix:
        #x = preprocessing.scale(x)  
        #col_name = 'word'+str(i);
        col_name = wordList[i]
        X_train[col_name] = x
        i = i+1
        
    y_train = pd.DataFrame({'popularity_score': X_pop})
    

     # Create another dataframe just for fun
#    fun = pd.DataFrame(
#         {#'text': X_text,
#          'is_root': X_root,
#          'controversiality': X_contro,
#          'children': X_chil,
#          'log_children': X_logchil,
#          'num_words': X_numwords, # new feature
#          'num_words_log': X_numwords_log,
#          #'sum_list': np.log([x+1 for x in sum_list]),
#
#
#          'words_avglen': X_avglen,
#          'words_avglen_log': X_avglen_log,
#
#          #'intersection_cardinality': X_intersect_card, # new feature
#          'popularity_score': X_pop
#         })
#    #visualize the relationship between the features and the response using scatterplots
#    #sns.pairplot(fun, x_vars=['is_root','controversiality','children','text_length','intersection_cardinality'], y_vars='popularity_score', height=7, aspect=0.7, kind='reg')
#    sns.pairplot(fun, x_vars=['is_root','controversiality','children','log_children', 'num_words', 'num_words_log',  'sum_list', 'words_avglen', 'words_avglen_log'], y_vars='popularity_score', height=7, aspect=0.7, kind='reg')

    return X_train, y_train

# =============================================================================
# Trainig Part
# =============================================================================

def training_runner(data, init_eta, init_beta, init_epsilon):

    X_train, y_train = data_preprocessing(data)
    # Least squares 
    start = time.time()
    err_ls, W_ls = least_squares_runner(X_train, y_train)
    end = time.time()
    print("Time using least squares:", end - start)

    # Gradient descent
    err_record_list = [] # store the process of how error change
    weight_list = []
    records = []
    for x in init_eta:
        for y in init_beta:
            for z in init_epsilon:
                start = time.time()
                W_gd, err_gd, err = gradient_descent_setup(X_train, y_train, x, y, z)
                records.append(dict(eta=x, beta=y, epsilon=z, train_err=err, weight=W_gd))
                err_record_list.append(err_gd)
                weight_list.append(W_gd)
                end = time.time()
                delta = end -start
                records[-1]['time'] = delta
                print("Time using gradient descent:", end-start)
    
    print("Error using least squares on training set:", err_ls)
    #print("Error using gradient descent on training set:", err)
    #print(W_gd)
    return W_ls, weight_list, records, err_record_list

# For 60 words
    
#init_eta = [0.00003] # initial learning rate (decrease to reduce instability & #iteration)
#init_beta = [0.007] # controls the speed of the decay (greater beta, faster converge)
#init_epsilon = [0.00002]
    
init_eta = [0.000020,0.000030,0.000040]
init_beta = [0.005,0.007,0.010]
init_epsilon = [0.000020,0.00030,0.00040]

# For 160 words
    
#init_eta = [0.000015] # initial learning rate (decrease to reduce instability & #iteration)
#init_beta = [0.01] # controls the speed of the decay (greater beta, faster converge)
#init_epsilon = [0.001]
    
#init_eta = [0.000010,0.000015,0.000020]
#init_beta = [0.010,0.050,0.100]
#init_epsilon = [0.0001,0.0010,0.0020]

W_ls, weight_list, records, err_record_list = training_runner(training, init_eta, init_beta, init_epsilon)

i = 0
for err_record in err_record_list:
    fig = plt.figure()
    plt.plot(err_record)
    name = "eta=" + str(records[i]['eta']) + " beta=" + str(records[i]['beta']) + " epsilon=" + str(records[i]['epsilon'])
    fig.suptitle(name, fontsize=16)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    i = i+1

# =============================================================================
# Validation Part
# =============================================================================

def validation_runner(data, W_list, records):
    X_val, y_val = data_preprocessing(data)

    i = 0
    for W in W_list:
        err = error_function(W, X_val, y_val)
        records[i]['validation_err'] = err
        i = i+1
    
    return records

records = validation_runner(validation, weight_list, records)

sorted_records = sorted(records, key=lambda k: k['validation_err']) # records sorted by time
records_copy = deepcopy(sorted_records)
for x in records_copy:
    del x['weight']
print(tabulate.tabulate(records_copy, headers={'eta': 'eta', 'beta': 'beta', 'epsilon': 'epsilon', 'train_err': 'train_err', 'time': 'time', 'validation_err': 'validation_err'}))

# =============================================================================
# Test Part
# =============================================================================

def testing_runner(data, W_ls, W_gd):

    X_test, y_test = data_preprocessing(data)
    # Least squares 
    err_ls = error_function(W_ls, X_test, y_test)

    # Gradient descent
    err_gd = error_function(W_gd, X_test, y_test)

    print("Error using least squares on testing set:", err_ls)
    print("Error using gradient descent on testing set:", err_gd)
    
    # Plot
    y_test = np.array(y_test.values.tolist())
    
    t = np.arange(0., len(y_test),1)
    fig = plt.figure()
    plt.plot(t, y_test, 'k.', t, np.dot(X_test, W_ls), 'r.')
    fig.suptitle("Closed-form prediction vs True value", fontsize=16)
    plt.xlabel('data points', fontsize=14)
    plt.ylabel('popularity score', fontsize=14)
    plt.show()
    
    fig = plt.figure()
    plt.plot(t, y_test, 'k.', t, np.dot(X_test, W_gd), 'r.')
    fig.suptitle("Gradient descent prediction vs True value", fontsize=16)
    plt.xlabel('data points', fontsize=14)
    plt.ylabel('popularity score', fontsize=14)
    plt.show()
       
    
    
W_gd = sorted_records[0]['weight']
testing_runner(test, W_ls, W_gd)



total_end = time.time()
print("Total running time is", total_end-total_start)















