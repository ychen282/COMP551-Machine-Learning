import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import time
import string

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 
start = time.time()

# Example:
data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
#for info_name, info_value in data_point.items():
#    print(info_name + " : " + str(info_value))

#split data into Training, Validation, Test sets
n=10000
train = data[:n]
validation = data[10000:11000]
test = data[11000:12000]

#function that implement the X matrix with data and features
def matrix(dataset):
    n = len(dataset)
    #append all the text
    append = ''
    for i in range(0,n):
        append += dataset[i]['text'].lower()+' '
    
    #split the appended text into words
    words = append.split()
      
    #count word frequency and store in a dictionary d
    d = {}
    for i in range(0, len(words)):
        key = words[i]
        if key in d:
            d[key] += 1
        else:
            d[key] = 1

    #find the 160 most frequente words
    d_160 = {}
    t = 60
    for i in range(1,t+1):
        key = ''
        temp = max(d, key = d.get)
        d_160[temp] = i
        d.pop(temp, None)
    print(d_160)
    
    #compute feature 1: frequent words
    feature_text = np.zeros((n,t))
    for i in range(0, n):
        temp = dataset[i]['text'].lower()
        temp_words = temp.split()
        l = len(temp_words)
        for j in range(0, l):
            key = temp_words[j]
            if key in d_160:
                feature_text[i,d_160[key]-1] += 1
  #              if j< l/3*2:
   #                 feature_text[i,d_160[key]-1] += 1
    #            else:
     #               feature_text[i,d_160[key]-1] += 2
                
    
    #feature 2: controversiality
    vector_cont = [0]*n
    for i in range(0, n):
        vector_cont[i] = dataset[i]['controversiality']
    
    #feature 3: children
    vector_child = [0]*n
    for i in range(0, n):
        vector_child[i] = dataset[i]['children']
        
    #feature 4: is_root
    vector_isRoot = [0]*n
    for i in range(0, n):
        temp = 0
        if dataset[i]['is_root'] == True:
            temp = 1
        vector_isRoot[i] = temp
        
    #feature 5: has question
    vector_ques = [0]*n
    for i in range(0,n):
        if dataset[i]['text'].find('?')!=-1:
            vector_ques[i] = 1
            
    #feature 6: emotion words
    vector_emo = [0]*n
    l_emo = ['like', 'didn’t', 'would', 'how', 'but', 'because', 'know', 'never', 'than', 'will', 'going', 'want', 'show', 'always', 'feel', 'fucking', 'fuck', 'actually', 'need', 'most', 'please', 'thought', 'same', 'should', 'last', 'made', 'well']
    for i in range(0,n):
        for j in range(0,len(l_emo)):
            if dataset[i]['text'].find(l_emo[j])!=-1:
                vector_emo[i] += 1
    
    #extract y
    vector_y = [0]*n
    for i in range (0, n):
        vector_y[i] = dataset[i]['popularity_score']
    
    #m is a matrix holding all features
    #insert dummy variables
    m = np.c_[feature_text, vector_cont, vector_child, vector_isRoot, vector_ques, [1]*n]
    
    return m, vector_y  

def training (dataset):
    (m, y) = matrix(dataset)
    #computation
    XTX = np.dot(np.transpose(m),m)
    XTX_inv = np.linalg.inv(XTX)
    XTX_inv_XT = np.dot(XTX_inv, np.transpose(m))
    w = np.dot(XTX_inv_XT, y)
    
    result = np.dot(m, w)
    error = np.dot(np.transpose((y - result)),(y-result))/len(y)
    print("training: close form error for training set: ", error)
    
    #initialize w
    w2 = [0]*len(m[0])
    j=0
    
    yita_set = [0.000015]
    beta_set = [0.0001]
    diff_set = [0.01]
    
    store_para = {}
    store_w = {}
    
    for yita in yita_set:
        for beta in beta_set:
            for diff in diff_set:
                #default diff
                d = 1000
                #iteration
                i = 1
                while d>diff:
                    w_temp = w2
                    alpha = yita/(1+beta*i)
                    delta = 2*alpha*(np.dot(XTX,w_temp)-np.dot(np.transpose(m),y))
                    w2 = w_temp - delta
                    w2 = w2/np.linalg.norm(w2)
                    d = np.linalg.norm(delta)
                    #print(d)
                    i=i+1
                para = [yita, beta, diff]
                result2 = np.dot(m, w_temp)
                error2 = np.dot(np.transpose((y - result2)),(y-result2))/len(y)
                print("training: current parameters yite beta diff are: ", para)
                print("training: current gradient descent error is", error2)
                store_para[j]=para
                store_w[j]=w_temp
                j=j+1
                
    return w, store_para

def validating (dataset, info):
    (w_set, para_set) = info
    (m, y) = matrix(dataset)
    w = w_set
    result = np.dot(m, w)
    e = np.dot(np.transpose((y - result)),(y-result))/len(y)
    
    temp_e = 100
    print("validation: best gradient descent error and best its are ", e, 0)
    return w

def testing(weight, dataset):
    (m, y) = matrix(dataset)
    result = np.dot(m, weight)
    error2 = np.dot(np.transpose((y - result)),(y-result))/len(y)
    return error2

print("test set: gradient descent error is ", testing(validating(validation, training(train)),test))

end = time.time()
print("Total time:", end-start)
    
    
    