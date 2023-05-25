import os
import numpy as np
import pandas as pd

import sys
sys.path.append('./Features')

from FeatureExtractor import FeaturesInfo

class data:
    def __init__(self, matrix, big_five, features = None) -> None:
        self.matrix = matrix
        self.big_five = big_five
        self.features = features

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

    #np.save('Dataset/error', npys[0][1])
    
    for i in range(len(xlsx)):
        for j in npys[i]:
            #dataset.append((j, xlsx.values[i]))
            #feats = get_features(j)
            print(len(dataset))
            a = FeaturesInfo(j)
            dataset.append(data(j, xlsx.values[i], a))
    return dataset

def main():
    dataset = build_dataset()
    arr = np.asanyarray(dataset, dtype=object)
    np.save('Dataset/dataset_feat', arr)
    
main()
