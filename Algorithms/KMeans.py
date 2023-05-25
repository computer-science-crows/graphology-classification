import numpy as np
from sklearn.datasets import load_digits


def k_means(dataset):
    data = []
    labels = []
    for i in range(len(dataset)):
        data.append(dataset[i][0])
        labels.append(dataset[i][1])
