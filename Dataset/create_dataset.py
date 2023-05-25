import os
import numpy as np
import pandas as pd


def get_xslx():
    cwd = os.getcwd()
    return pd.read_excel(cwd+'/Scrapper/big_five.xlsx')
    # return pd.read_excel('D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Scrapper/big_five.xlsx')


def get_images():
    cwd = os.getcwd()
    path = cwd+'/Image_Matrix/Img_npy'

    # path = 'D:/UniVerSiDaD/IV Ano/Machine Learning/Project/Graphology classification/graphology-classification/Image_Matrix/Img_npy'

    folders_names = os.listdir(path)

    arr = []
    for i in folders_names:
        arr.append(int(i))
    arr.sort()

    npys = {}

    for i, name in enumerate(arr):
        img_path = path + '/' + str(name)
        imgs_names = os.listdir(img_path)

        imgs_names_arr = []
        for k in imgs_names:
            dot_pos = k.find('.', 0, len(k))
            number = k[0:dot_pos]
            imgs_names_arr.append(int(number))
        imgs_names_arr.sort()

        for j in imgs_names_arr:
            if (i not in npys):
                npys[i] = []
            npys[i].append(np.load(img_path + '/' + str(j) + '.npy'))

    return npys


def build_dataset():
    xlsx = get_xslx()
    npys = get_images()
    dataset = []

    for i in range(len(xlsx)):
        for j in npys[i]:
            dataset.append((j, xlsx.values[i]))

    return dataset


dataset = build_dataset()
