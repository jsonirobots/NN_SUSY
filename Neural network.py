#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras  ')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install numpy')
get_ipython().system('pip install uproot')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install tensorboard')


# In[9]:


from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import pandas as pd
import math
import numpy as np
import tensorflow as tf
import datetime

from tensorflow.keras.callbacks import TensorBoard
get_ipython().run_line_magic('load_ext', 'tensorboard')



df= pd.read_csv('/Users/ryanrushing/Desktop/SUSY.txt', names=['index','phi1','phi2','eta1','eta2','diphi','deta','deltaR','genweight','recoweight'])
susydatanormalized = df.copy()
susydatanormalized.drop(columns='index',inplace=True)
susydatanormalized.drop(columns='genweight',inplace=True)
susydatanormalized.drop(columns='recoweight',inplace=True)
phi1='phi1'
susydatanormalized[phi1] = susydatanormalized[phi1] /math.pi
phi2='phi2'
susydatanormalized[phi2] = susydatanormalized[phi2] /math.pi
eta1='eta1'
susydatanormalized[eta1] = susydatanormalized[eta1] /susydatanormalized[eta1].abs().max()
eta2='eta2'
susydatanormalized[eta2] = susydatanormalized[eta2] /susydatanormalized[eta2].abs().max()
diphi='diphi'
susydatanormalized[diphi] = susydatanormalized[diphi] /susydatanormalized[diphi].abs().max()
deta='deta'
susydatanormalized[deta] = susydatanormalized[deta] /susydatanormalized[deta].abs().max()
deltaR='deltaR'
susydatanormalized[deltaR] = susydatanormalized[deltaR] /susydatanormalized[deltaR].abs().max()
identifier=[1] * 6760
susydatanormalized['identifier']=identifier

df2= pd.read_csv('/Users/ryanrushing/Desktop/ttbarsignalplustau_mainSignal.txt', names=['index','phi1','phi2','eta1','eta2','diphi','deta','deltaR','genweight','recoweight'])
standarddatanormalized = df2.copy()
standarddatanormalized=standarddatanormalized[:6760]
standarddatanormalized.drop(columns='index',inplace=True)
standarddatanormalized.drop(columns='genweight',inplace=True)
standarddatanormalized.drop(columns='recoweight',inplace=True)
phi1='phi1'
standarddatanormalized[phi1] = standarddatanormalized[phi1] /math.pi
phi2='phi2'
standarddatanormalized[phi2] = standarddatanormalized[phi2] /math.pi
eta1='eta1'
standarddatanormalized[eta1] = standarddatanormalized[eta1] /standarddatanormalized[eta1].abs().max()
eta2='eta2'
standarddatanormalized[eta2] = standarddatanormalized[eta2] /standarddatanormalized[eta2].abs().max()
diphi='diphi'
standarddatanormalized[diphi] = standarddatanormalized[diphi] /standarddatanormalized[diphi].abs().max()
deta='deta'
standarddatanormalized[deta] = standarddatanormalized[deta] /standarddatanormalized[deta].abs().max()
deltaR='deltaR'
standarddatanormalized[deltaR] = standarddatanormalized[deltaR] /standarddatanormalized[deltaR].abs().max()
identifier=[0] * 6760
standarddatanormalized['identifier']=identifier

frames = [standarddatanormalized,susydatanormalized]
wholedata = pd.concat(frames)
wholedata=wholedata.to_numpy()
np.random.shuffle(wholedata)

trainx=wholedata[:(round(0.8*len(wholedata[:,2])))]
testx=wholedata[:(round(0.2*len(wholedata[:,2])))]

phi1x,phi2x,eta1x,eta2x,dihpix,detax,deltaRx,identifierx=np.hsplit(trainx,8)
trainX=np.concatenate((phi1x,phi2x,eta1x,eta2x,dihpix,detax,deltaRx),axis=1)

trainY=identifierx
phi1y,phi2y,eta1y,eta2y,dihpiy,detay,deltaRy,identifiery=np.hsplit(testx,8)
testX=np.concatenate((phi1y,phi2y,eta1y,eta2y,dihpiy,detay,deltaRy),axis=1)
testY=identifiery

print('trainx shape',trainX.shape)
print('trainy shape',trainY.shape)
print('testx shape',testX.shape)
print('testy shape',testY.shape)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# load train and test dataset
#def load_dataset():
 #load dataset
#(trainX, trainY), (testX, testY) = (trainx,trainy),(testx,testx)
 #one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)
#return trainX, trainY, testX, testY
 
def define_model():
    model = Sequential()
    model.add(Dense(7,activation='relu',input_dim=7))
    model.add(Dense(14, activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(42, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(21, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='sigmoid'))
    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
 
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model 
        history = model.fit(trainX, trainY, epochs=400,batch_size=10816, validation_data=(testX, testY), verbose=0,callbacks=[tensorboard_callback])
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories



 
# run the test harness for evaluating a model
def run_test_harness():
#load dataset
    #trainX, trainY, testX, testY = load_dataset()

# evaluate model
    scores, histories = evaluate_model(trainX, trainY)
# learning curves
    #summarize_diagnostics(histories)
# summarize estimated performance
    #summarize_performance(scores)

    
 
  
get_ipython().run_line_magic('rm', '-rf ./logs/')
run_test_harness()
get_ipython().run_line_magic('reload_ext', 'tensorboard')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




