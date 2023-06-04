from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from dts import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

def group_10(big_five):
    big_five = np.array(big_five)
    big_five[big_five <=10] = 130
    big_five[big_five <= 20] = 2
    big_five[big_five == 130] = 0
    big_five[big_five >110] = 11
    big_five[big_five >100] = 10
    big_five[big_five > 90] = 9
    big_five[big_five > 80] = 8
    big_five[big_five >70] = 7
    big_five[big_five >60] = 6
    big_five[big_five >50] = 5
    big_five[big_five >40] = 4
    big_five[big_five >30] = 3
    big_five[big_five >20] = 1
    return big_five

df = np.load('Dataset/dataset_feat.npy', allow_pickle=True)
tr = np.load('Dataset/train.npy', allow_pickle=True)
te = np.load('Dataset/test.npy', allow_pickle=True)

matrix = [i.matrix for i in tr]
matrix2 = [i.matrix for i in te]
matrix.extend(matrix2)
matrix = np.array(matrix)


res = [i.big_five for i in tr]
res2 = [i.big_five for i in te]
res.extend(res2)
res = np.array(res)

res = group_10(res)

def resizing(matrix, res, x, y):
    mtz = np.array([cv2.resize(i, (x, y)).astype('int64') for i in matrix])
    xtrain, xtest, ytrain, ytest = train_test_split(mtz, res, test_size = 0.30, random_state=23)
    return xtrain, xtest, ytrain, ytest

def cnn(x, y, xtrain, ytrain, xtest, ytest):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(x, y, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='sigmoid')) # 5 neuronas de salida para 5 etiquetas posibles

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(xtrain, ytrain, epochs=10, batch_size=32)
    
    print('Evaluation results')
    model.evaluate(xtest, ytest)

def run(mtz, res, x, y):
    xtrain, xtest, ytrain, ytest = resizing(mtz, res, x, y)
    cnn(x, y, xtrain, ytrain, xtest, ytest)
    
run(matrix, res, 64, 64)