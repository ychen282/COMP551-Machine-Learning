from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,MaxPooling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras import optimizers
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',input_shape=(64,64,1),activation='relu'))#16
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))#16
#model.add(Conv2D(filters=32,kernel_size=(5,5),padding='same',activation='relu'))#16
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
#model.add(Dropout(0.25))#0.25

model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu'))#16
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=128,kernel_size=(5,5),padding='same',activation='relu'))#36
model.add(Conv2D(filters=128,kernel_size=(5,5),padding='same',activation='relu'))#duo
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))#0.25


model.add(Conv2D(filters=256,kernel_size=(5,5),padding='same',activation='relu'))#36
model.add(Conv2D(filters=256,kernel_size=(5,5),padding='same',activation='relu'))#duo
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.5))#0.25

model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))


model.add(Dense(10,activation='softmax'))
print(model.summary())



# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),       
#                  activation='relu',padding='same',
#                  input_shape=(64,64,1)))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# print(model.summary())

import numpy as np # linear algebra
#np.set_printoptions(threshold=np.nan)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import csv
import pickle

# print(os.listdir("./input"))
np.random.seed(10)

import pandas as pd
train_images_raw = pd.read_pickle('./input/train_images.pkl')
train_labels_raw = pd.read_csv('./input/train_labels.csv')
test_images_raw = pd.read_pickle('./input/test_images.pkl')
# print("train_images_raw.shape: ",train_images_raw.shape)
# print("train_images_raw: ", train_images_raw)

# print("train_labels_raw.shape: ",train_labels_raw.shape)




# print(train_images_raw[0][0])
# print(train_images_raw[0][0].shape)



# print(train_images_raw[0])
# print(train_images_raw[0].shape)


#-------------------------------------------------------------
from PIL import Image
import matplotlib.pyplot as plt


import cv2
from scipy.ndimage.measurements import label

#for img_idx in range(0,20):#(2,6): 2,3,4,5


for i in range(len(train_images_raw)): #40000 images
    img=train_images_raw[i]	

    for line in range(64):
       for pixel in range(64):
            if (img[line][pixel]<255):
                img[line][pixel]=0

    #print(img)
    #print(img.shape)




#array = np.array(...)  # your example

    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter

    # structure  # in this case we allow any kind of connection
    # Out[5]: 
    # array([[1, 1, 1],
    #        [1, 1, 1],
    #        [1, 1, 1]])

    labeled, ncomponents = label(img, structure)

    #print(labeled)
    #print(ncomponents)

    
    #my_matrix = numpy.loadtxt(open("./","rb"),delimiter=",",skiprows=0)
    #numpy.savetxt('./Array/processed.csv', labeled, delimiter = ',')


    unique, counts = np.unique(labeled, return_counts=True)

    #print(unique, counts)

    zipped=dict(zip(unique, counts))

    maxi=0
    for key in zipped:
        if (key!=0):
            if zipped[key]>maxi:
                maxi= zipped[key]
                maxkey=key

    #print('maxi:', maxi)
    #print(maxkey)


    indices = np.indices(img.shape).T[:,:,[1, 0]]
    position=indices[labeled == maxkey]

    #print('len position:', len(position))
    #print(position)
    #print(position[0][0])

    outputarr=np.zeros(shape=(64,64))
    #print(len(x))

    for i in range(len(position)):
        #print(position[i][0], position[i][1])
        x=position[i][0]
        y=position[i][1]
        # print('x: ', x)
        # print('y: ', y)
        outputarr[x][y]=255

    train_images_raw[i] = outputarr

print("train_images_raw after process: ", train_images_raw)


output = open('train_images_raw.pkl', 'wb')
pickle.dump(train_images_raw, output)
#---------------------------------------------------------------------
pkl_file = open('train_images_raw.pkl', 'rb')
train_images_raw = pickle.load(pkl_file)


print("reloading the train_images_raw.pkl file...")
# print(train_images_raw)
# print(train_images_raw.shape)
pkl_file.close()
#==============================================================================

for i in range(len(test_images_raw)): #40000 images
    img=test_images_raw[i]	

    for line in range(64):
       for pixel in range(64):
            if (img[line][pixel]<255):
                img[line][pixel]=0

    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter

    labeled, ncomponents = label(img, structure)

    unique, counts = np.unique(labeled, return_counts=True)

    #print(unique, counts)

    zipped=dict(zip(unique, counts))

    maxi=0
    for key in zipped:
        if (key!=0):
            if zipped[key]>maxi:
                maxi= zipped[key]
                maxkey=key
    indices = np.indices(img.shape).T[:,:,[1, 0]]
    position=indices[labeled == maxkey]

    outputarr=np.zeros(shape=(64,64))
    #print(len(x))

    for i in range(len(position)):
        #print(position[i][0], position[i][1])
        x=position[i][0]
        y=position[i][1]
        outputarr[x][y]=255

    test_images_raw[i] = outputarr

print("test_images_raw after process: ", test_images_raw)

output = open('test_images_raw.pkl', 'wb')
pickle.dump(test_images_raw, output)
#---------------------------------------------------------------------
pkl_file = open('test_images_raw.pkl', 'rb')
test_images_raw = pickle.load(pkl_file)

print("reloading the test_images_raw.pkl file...")
# print(test_images_raw)
# print(test_images_raw.shape)
pkl_file.close()

#==============================================================================



t_lables=[0]*len(train_labels_raw)


#print(train_labels_raw)
# 	t_lables[i]=train_labels_raw[i][1]

for i in range(len(train_labels_raw)):
    t_lables[i]=train_labels_raw.at[i, 'Category']
#print(t_lables)

x_Train4D =train_images_raw.reshape(40000,64,64,1).astype('float32')


x_Real_Test=test_images_raw.reshape(10000,64,64,1).astype('float32')

#print(x_Train4D)
print ('x_train:',x_Train4D.shape)
#print ('x_test:',x_Test4D.shape)

x_Train4D_normalize = x_Train4D/ 255
# x_Test4D_normalize = x_Test4D/ 255
x_Real_Test_normalize=x_Real_Test/ 255

#print(x_Train4D_normalize)
#print(type(x_Train_normalize))
#print(y_train_lable[:5])
#print('y_train_lable: ',y_train_label)

#print('-------------------------------------')

# y_TrainOneHot = np_utils.to_categorical(y_train_label)
# y_TestOneHot = np_utils.to_categorical(y_test_label)



y_TrainOneHot = np_utils.to_categorical(t_lables)
# y_TestOneHot = np_utils.to_categorical(y_test_label)
#print(y_TrainOneHot[:3])


batch_size = 100
epochs = 25

#checkpointer_1 = ModelCheckpoint(filepath="weights-{epoch:02d}.hdf5", verbose=1, save_best_only=False, period=10)##


#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#sgd = optimizers.SGD(lr=0.0001, momentum=0.9, decay=0, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.00001)
#model.fit(X_train, Y_train, callbacks=[reduce_lr])


mc = ModelCheckpoint('best_model.hdf5', monitor='val_loss', mode='min', save_best_only=True)


#train_history = model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[checkpointer_1])##
#model.save_weights('cnn_drop4096_weights.h5')###
train_history = model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.1,
             epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[reduce_lr,mc])##





import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')


prediction = model.predict_classes(x_Real_Test_normalize)
print("prediction 1:", prediction)
prediction = pd.DataFrame(prediction, columns=['Category']).to_csv('prediction1_last.csv')

#-------------------------------------------------------------------------


print("Created model and loaded weights from file")
model.load_weights("best_model.hdf5")
# Compile model (required to make predictions)
#sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0004, nesterov=False)
#adam=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# load pima indians dataset


# scores = model.evaluate(x_Test4D_normalize, y_TestOneHot, verbose=1)




# print()
# print('loss=',scores[0])
# print('accuracy=',scores[1])

prediction = model.predict_classes(x_Real_Test_normalize)
print("prediction 2: ", prediction)
prediction = pd.DataFrame(prediction, columns=['Category']).to_csv('prediction2_lowest_loss.csv')

# seq_num = range(len(x_Real_Test))
# with open('predic2.csv','w+') as predict_writer:
#     predict_writer.writelines('Id,Label\n')
#     for test_num in seq_num:          
#         prediction = prediction[0].tolist()
#         label = reader.num_list[prediction.index(max(prediction))]
#         predict_writer.writelines(str(test_num+1) + ',' + str(label) + '\n')


# with open('cnm.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerows([['Id','Category']])

 
#     for i in range(10000):
#          writer.writerows([[i,prediction[i]]])





