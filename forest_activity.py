# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_yaml
from keras.callbacks import ModelCheckpoint
from keras import utils
import matplotlib.pyplot as plt
import librosa
import librosa.display 
import numpy as np
import pandas as pd
import random
import warnings 

warnings.filterwarnings('ignore')
def load_dataset(path,file_name,fold,category, category_id):
    data = pd.read_csv(path)
    data.head(5)
    data.shape
    valid_data = data[[file_name,fold,category_id,category]]
    valid_data.shape
    return data, valid_data

def convert_to_spectrogram(path):
    y, sr = librosa.load(path,duration=2.97) #"ESC-50-master/audio/1-137-A-32.wav"
    ps = librosa.feature.melspectrogram(y=y,sr=sr)
    ps.shape
    return ps,y
def display_spectogram(ps,y,path):
    ps,y=convert_to_spectrogram(path)#"ESC-50-master/audio/1-137-A-32.wav"
    librosa.display.specshow(ps,y_axis='mel',x_axis='time')

def generate_dataset(path, valid_data,file_name):
    D=[] #dataset
    valid_data['path'] = valid_data[file_name].astype('str')
    for row in valid_data.itertuples():
        ps,y= convert_to_spectrogram(path+'/'+row.path)
        if ps.shape != (128, 128): 
            continue
        D.append((ps,row.target))
        
    return D

def data_normalization(dataset):
    random.shuffle(dataset)
    train = dataset[:1600]
    test = dataset[1600:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    X_train = np.array([x.reshape((128,128,1)) for x in X_train])
    X_test = np.array([x.reshape((128, 128, 1)) for x in X_test])
    y_train = np.array(utils.to_categorical(y_train, 50))
    y_test = np.array(utils.to_categorical(y_test, 50))
    return X_train,X_test,y_train,y_test

def baseline_model():
    model = Sequential()
    input_shape = (128, 128, 1)
    model.add(Conv2D(24,(5,5), strides=(1,1),input_shape=input_shape))
    model.add(MaxPooling2D((4,2),strides=(4,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48,(5,5), padding="valid"))
    model.add(MaxPooling2D((4,2),strides=(4,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48,(5,5), padding="valid"))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(50))
    model.add(Activation('softmax'))
    model.compile(
        optimizer="Adam",
        loss ="categorical_crossentropy",
        metrics=['accuracy']
        )
    return model

def train_model(model,X_train, X_test,y_train,y_test,callbacks_list):
    history=model.fit( X_train,y_train,epochs=12,batch_size=128, validation_data=(X_test,y_test),callbacks=callbacks_list,verbose=0  )
    return history

def evaluation(X_test, y_test):
    score = model.evaluate(x=X_test, y=y_test)
    return score

def save_model(model, path):
    model_yaml = model.to_yaml()
    with open(path, "w") as yaml_file:
        yaml_file.write(model_yaml)   
    model.save_weights("model.h5")
    
def load_model(path):
    yaml_file = open(path, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    #load weights
    loaded_model.load_weights(path)
    return loaded_model

def checkpoint(path):
    checkpoint=ModelCheckpoint(path, monitor='val_acc',verbose=1,save_best_only=True, mode='max')
    callbacks_list=[checkpoint]
    return callbacks_list

def display_history(history, evtype):
    plt.plot(history.history[evtype.astype('str')])
    plt.plot(history.history['val'+evtype.astype('str')])
    if evtype.astype('str')=='acc':
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
    else:
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        
    plt.legend(['train','test'], loc='upper left')
    

data,valid_data=load_dataset("ESC-50-master/meta/esc50.csv",'filename','fold','category', 'target')

D = generate_dataset("ESC-50-master/audio", valid_data,'filename')
X_train, X_test,y_train,y_test = data_normalization(D)
model = baseline_model()
callbacks_list=checkpoint('weight.best.hdf5')
train_model(model,X_train, X_test,y_train,y_test,callbacks_list)
score=evaluation(X_test, y_test)

print('Test Loss:', score[0])
print('Test accuracy:',score[1])








 