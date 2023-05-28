import numpy as np
from sklearn.cluster import KMeans
from dts import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def k_means():
    ds = np.load('Dataset/dataset_feat.npy', allow_pickle=True)
    data = []

    for i in range(len(ds)):
        f = ds[i].features
        data.append([f.slant, f.baseline, f.word_space,
                    f.line_space, f.margin])

    kmeansModel = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data)
    for i in range(len(data)):
        print(kmeansModel.labels_[i])

    # elbow_method(data)


def elbow_method(data):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 11)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(data)
        kmeanModel.fit(data)

        d = sum(np.min(cdist(data, kmeanModel.cluster_centers_,
                'euclidean'), axis=1)) / len(data)
        i = kmeanModel.inertia_

        distortions.append(d)
        mapping1[k] = d

        inertias.append(i)
        mapping2[k] = i

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()


def plot_results(data):
    pass


k_means()
