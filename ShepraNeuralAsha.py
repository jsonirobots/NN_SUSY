import os
import tempfile
from keras.models import load_model
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
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
import sherpa
import shutil


def loaddataset():
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
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
def main():
    trainX, trainY, testX, testY = loaddataset()
    parameters = [sherpa.Continuous(name='lr', range=[0.005,.1]),
                  sherpa.Discrete(name="num_of_units_2",range=[14,56])]

    algorithm = alg = sherpa.algorithms.SuccessiveHalving(r=10, R=90, eta=3, s=0, max_finished_configs=5)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False, dashboard_port=None, disable_dashboard=False)
    model_dir = tempfile.mkdtemp()
    for trial in study:
        # Getting number of training epochs
        initial_epoch = {10: 0, 30: 1, 90: 4}[trial.parameters['resource']]
        epochs = trial.parameters['resource'] + initial_epoch

        print("-" * 100)
        print(f"Trial:\t{trial.id}\nEpochs:\t{initial_epoch} to {epochs}\nParameters:{trial.parameters}\n")

        if trial.parameters['load_from'] == "":
            print(f"Creating new model for trial {trial.id}...\n")

            # Get hyperparameters
            lr = trial.parameters['lr']
            num_units = trial.parameters['num_of_units_2']


            # Create model
            model = Sequential()
            model.add(Dense(7, activation='relu', input_dim=7))
            model.add(Dense(14, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(num_units, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(14, activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(2, activation='sigmoid'))
            opt = SGD(lr=lr, momentum=0.5)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        else:
            print(f"Loading model from: ", os.path.join(model_dir, trial.parameters['load_from']), "...\n")

            # Loading model
            model = load_model(os.path.join(model_dir, trial.parameters['load_from']))
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        for i in range(initial_epoch, epochs):
            model.fit(trainX, trainY, initial_epoch=i, epochs=i + 1)
            loss, accuracy = model.evaluate(testX, testY)

            print("Validation accuracy: ", accuracy)
            study.add_observation(trial=trial, iteration=i,
                                  objective=accuracy,
                                  context={'loss': loss})

        study.finalize(trial=trial)
        print(f"Saving model at: ", os.path.join(model_dir, trial.parameters['save_to']))
        model.save(os.path.join(model_dir, trial.parameters['save_to']))

        study.save(model_dir)

    print(study.get_best_result())
    shutil.rmtree(model_dir)
    input("Type enter to End the code")
main()
