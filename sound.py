# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:17:14 2019

@author: Degaga
"""


from memory_profiler import memory_usage
from keras import layers
from keras import models
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras_preprocessing.image import ImageDataGenerator
from keras.models import model_from_yaml
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib import figure
import librosa
import librosa.display
import pandas as pd
from glob import glob
import numpy as np
import gc

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'train/' + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'test/' + name + '.png'
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
def generate_csv(path):
    file = pd.read_csv(path)
    train_file = file[:1600]
    test_file = file[1600:]
    train=[]
    test=[]
    for row in train_file.itertuples():
        wav = row.filename
        png = wav.split('.')[0]
        train.append(png + '.png')
    train_file['filename'] = train
    
    for row in test_file.itertuples():
        wav = row.filename
        png = wav.split('.')[0]
        test.append(png + '.png')
        
    test_file['filename'] = test
    return train_file, test_file

Data_dir=np.array(glob("ESC-50-master/audio/*"))
i=0
for file in Data_dir[i:i+1600]:
    #Define the filename as is, "name" refers to the JPG, and is split off into the number itself. 
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()

i=1600
for file in Data_dir[i:]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()

train_file, test_file=generate_csv('ESC-50-master/meta/esc50.csv')
train_file.to_csv('dataset/train.csv')
test_file.to_csv('dataset/test.csv')

def append_ext(fn):
    return fn+".png"

def generate_data(train_csv_path, test_csv_path, train_image_path, test_image_path):
    traindf=pd.read_csv(train_csv_path,dtype=str)
    testdf=pd.read_csv(test_csv_path,dtype=str)
    traindf["id"]=traindf["target"].apply(append_ext)
    testdf["id"]=testdf["target"].apply(append_ext)
    
    datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
    
    train_generator=datagen.flow_from_dataframe(
            dataframe=traindf,
            directory=train_image_path,
            x_col="filename", y_col="category",
            subset="training", batch_size=32,  seed=42,
            shuffle=True, class_mode="categorical",
            target_size=(128,128))
    valid_generator=datagen.flow_from_dataframe(
            dataframe=traindf, directory=train_image_path,
            x_col="filename",  y_col="target", subset="validation",
            batch_size=32,  seed=42,  shuffle=True,  class_mode="categorical",
            target_size=(128,128))
    test_datagen=ImageDataGenerator(rescale=1./255.)
    test_generator=test_datagen.flow_from_dataframe(
            dataframe=testdf,  
            directory= test_image_path,
            x_col="filename", y_col=None,
            batch_size=32, seed=42, shuffle=False,
            class_mode=None, target_size=(64,64))
   
    return train_generator, valid_generator, test_generator

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='softmax'))
    model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

def train_model(model, train_generator,valid_generator,callbacks_list):
    #Fitting keras model, no test gen for now
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=150,callbacks=callbacks_list)
    return history

def evaluation(model, valid_generator):
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    score = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)
    return score

def test_model(test_generator, train_generator):
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    test_generator.reset()
    pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST,verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)
    #Fetch labels from train gen for testing
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    print(predictions[0:6])

def save_model(model, path):
    model_yaml = model.to_yaml()
    with open(path+'model.yaml', "w") as yaml_file:
        yaml_file.write(model_yaml)   
    model.save_weights(path+"weights.h5")
    
def load_model(path):
    yaml_file = open(path+'model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    #load weights
    loaded_model.load_weights(path+'weight.hdf5')
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
def predict_class(model, weight, data):
    pred = model.predict_class()
    print(pred) 
    
model = baseline_model()
model.summary()

train_generator, valid_generator, test_generator= generate_data(
        'dataset/train.csv','dataset/test.csv',
        'dataset/train/audio','dataset/test/audio')

callbacks_list = checkpoint('weight.best.hdf5')
history = train_model(model, train_generator,valid_generator,callbacks_list)
score = evaluation(model, valid_generator)
save_model(model, 'saved_mdel')

display_history(history, 'acc')
display_history(history, 'loss')


