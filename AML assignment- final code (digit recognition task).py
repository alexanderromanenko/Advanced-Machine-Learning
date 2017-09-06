# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:33:23 2017

@author: alexr
"""
#loading required libraries
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import datasets

"""
Loading the data
"""
#loading the training set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# feature matrix
X = train.ix[:,1:]
X_train= X.values

# target vector
y = train['label']
y_train= y.values


"""
Analysis of the dataset
"""
#code for viewing the digits
def digit_view_train(i):
    img=train.iloc[:,1:].iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train['label'].iloc[i])

def digit_view_test(i):
    img=test.iloc[:,1:].iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(test['label'].iloc[i])


#code for demonstrating various ways of writing digits - showing 5 x 10 digits in a row
f, mtrx = plt.subplots(5,10,sharex = True,sharey=True)
plt.locator_params (nbins= 1, tight=True)
for i in range (0,10):
     for j in range (0,10):
        data= train[train['label']==j].iloc[i,1:].values
        nrows, ncols =28,28
        grid = data.reshape((nrows,ncols))
        n=math.ceil (j)
        m=[0,1,2,3,4]*3
        mtrx [m[i-1], n].imshow(grid, cmap='gray')

#distribution of numbers
plt.hist(train["label"])
plt.title("Frequency Histogram of Digits in Training Data   ")
plt.xlabel("Digit Value (0-9)")
plt.ylabel("Frequency")


"""
K- Nearest neighbour classifier
"""

knn = KNeighborsClassifier()

k_range = list(range(1, 100))
weight_options = ['uniform', 'distance']
param_dist = dict(n_neighbors=k_range, weights=weight_options)
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, n_jobs = 1, verbose=1)

rand.fit(X, y)

print(rand.best_score_)
print(rand.best_params_)

#running the classifier on the test data by using the best parameters 
knn = KNeighborsClassifier(n_neighbors=rand.best_params_['n_neighbors'], weights=rand.best_params_['weights'])
knn.fit(X, y)

predictions = knn.predict(test)

#saving the results as a csv file
result = pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
result.to_csv('knn result2.csv', index=False, header=True)

#STORING RESULTS OF Cross Validation FOR DIFFERENT K in order to establish an optimal K
#this code is different to the one above as I wanted to simplify it and run it for larger number of K in order to build a diagram of the accuracy vs K.
knn_results=[]
knn = KNeighborsClassifier()
for k in range (1,1000,10): 
    k_range= [k,k]
    print (k_range)    
    weight_options = ['uniform', 'distance']
    param_dist = dict(n_neighbors= k_range, weights=weight_options)
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=1, n_jobs=-1, verbose=1)
    rand.fit(X, y)
    knn_results.append(rand.best_score_)
    print (k, rand.best_score_)
knn_acc = pd.DataFrame (np.array ([list(range (1,len(knn_results)+1)), knn_results]).T,columns=["k","Accuracy"])

"""
SVM (no PCA)
"""

#normalisation /255
X4 = train.iloc[:,1:]/255
# target vector- label
y4 = train.ix[:,'label']

clf = SVC()
scores = cross_val_score(clf, X4, y4, cv=10)

svc_full=SVC()
svc_full.fit(X4,y4)

test4= test/255
predict_svm=svc_full.predict(test4)

#training SVM classifier
result = pd.DataFrame({'ImageId': list(range(1,len(predict_svm)+1)), 'Label': predict_svm})
result.to_csv('svm only(no pca) result v2.csv', index=False, header=True)

"""
Prnicpal Component Analysis
"""
#cross validation using PCA

#finding best n_components

cv_pca_best_n = dict()
y4 = train['label']
for n in range(1,101): 
    X_temp = train.iloc[:,1:]
    pca= PCA(n_components=n, whiten= True)
    pca.fit(X_temp)
    X_temp= pca.transform(X_temp)
    clf_pca_temp = SVC()
    scores_pca2 = cross_val_score(clf_pca_temp, X_temp, y4, cv=10)
    cv_pca_best_n[n]=scores_pca2.mean()
    
#highest accuracy achieved using n_components=42    


#plugging in the found n_components to transform the data and to use SVM for traing the model and prediction
#feature matrix
X4 = train.iloc[:,1:]
pca= PCA(n_components=42, whiten= True)
pca.fit(X4)
X4= pca.transform(X4)

# target vector- label
y4 = train['label']

#training SVM classifier
svc_pca=SVC()
svc_pca.fit(X4,y4)

test4= pca.transform(test)
predict_svm_pca=svc_pca.predict(test4)

result = pd.DataFrame({'ImageId': list(range(1,len(predict_svm_pca)+1)), 'Label': predict_svm_pca})
result.to_csv('SVM result (n=42) (PCA).csv', index=False, header=True)

## PCA decomposition- to be used in 'Variance Explained' graphs
pca = decomposition.PCA(n_components=200) #Finds first 200 PCs
pca.fit(X4)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of variance explained')
plt.xlabel('components')


"""
Convolutional Neural Network
"""
from subprocess import check_output
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
%matplotlib inline


#creating array for the training and test data
training_set = "train.csv"
test_set = "test.csv"
train_dataset = np.loadtxt(training_set, skiprows=1, dtype='int', delimiter=',')

#division of the training set into 2: training (85%) and validation sets (15%)
val_split = 0.15
n_raw = train_dataset.shape[0]
n_val = int(n_raw * val_split + 0.5)
n_train = n_raw - n_val

#shuffling the entries before the division into two sets
np.random.shuffle(train_dataset)
X_val, X_train = train_dataset[:n_val,1:], train_dataset[n_val:,1:]
y_val, y_train = train_dataset[:n_val,0], train_dataset[n_val:,0]

#normalisation of the data
X_train = X_train.astype("float32")/255.0
X_val = X_val.astype("float32")/255.0
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

#reshaping the data into 28 by 28 matrix
n_classes = y_train.shape[1]
X_train = X_train.reshape(n_train, 28, 28, 1)
X_val = X_val.reshape(n_val, 28, 28, 1)

#seting up the model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#just in time image generator to enforce the learning by providing newly generated images (by using rotation, zoom and position shifts)
datagen = ImageDataGenerator(zoom_range = 0.1, height_shift_range = 0.1, width_shift_range = 0.1, rotation_range = 20)
                            
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-3), metrics = ["accuracy"])

#setting up parameters for early stop
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto'), ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0)]

#runing the model
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 64), steps_per_epoch = 10000, epochs = 25, verbose = 1, validation_data = (X_val, y_val), callbacks = callbacks)


test_dataset = np.loadtxt(test_set, skiprows=1, dtype='int', delimiter=',')
X_test = test_dataset.astype("float32")/255.0
n_test = X_test.shape[0]
X_test = X_test.reshape(n_test, 28, 28, 1)
y_test = model.predict(X_test, batch_size=64)
y_index = np.argmax(y_test,axis=1)

#saving the restults as CSV file
result = pd.DataFrame({'ImageId': list(range(1,len(y_index)+1)), 'Label': y_index})
result.to_csv('CNN result (longer run).csv', index=False, header=True)


#displaying results of the model
history_dict= hist.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()