from dts import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import silhouette_visualizer

# global parameters
seed = 45
nk = 14
k = 7
cwd = os.getcwd() + '/Algorithms/Kmeans_plots/'


def k_means():
    ds = np.load('Dataset/dataset_feat.npy', allow_pickle=True)
    data = []
    features = []

    for i in range(len(ds)):
        f = ds[i].features
        features.append([f.slant, f.baseline, f.word_space,
                         f.line_space, f.margin])

    for i in range(len(ds)):
        bf = ds[i].big_five
        data.append(bf)

    # uncomment these lines to cluster according to big five personality traits in dataset
    # cwd += f'data=bigfive/seed{seed}/nk{nk-1}/'

    # elbow_method(data, nk, seed)
    # shilouette_method(data, nk, seed)

    # plot_results_2d(data, k, seed)
    # plot_results_3d(data, k, seed)

    # infered_results(data, features, k, seed)

    # ----------------------------------------------------------------------------------- #
    # uncomment these lines to cluster according to handwriting features in dataset
    # cwd += f'data=features/seed{seed}/nk{nk-1}/'

    # elbow_method(features, nk, seed)
    # shilouette_method(features, nk, seed)

    # plot_results_2d(features, k, seed)
    # plot_results_3d(features, k, seed)

    infered_results(features, data, k, seed)


def elbow_method(data, nk, seed):
    distortions = []
    inertias = []
    clusters_number = range(2, nk)

    for k in clusters_number:
        kmeanModel = KMeans(n_clusters=k, n_init='auto',
                            random_state=seed).fit(data)

        d = sum(np.min(cdist(data, kmeanModel.cluster_centers_,
                'euclidean'), axis=1)) / len(data)
        i = kmeanModel.inertia_

        distortions.append(d)
        inertias.append(i)

    plt.plot(clusters_number, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig(cwd + 'ElbowM_Distortion')
    plt.show()

    plt.plot(clusters_number, inertias, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.savefig(cwd + 'ElbowM_Inertia')
    plt.show()


def shilouette_method(data, nk, seed):
    sil = []
    clusters_number = range(2, nk)

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in clusters_number:
        kmeans = KMeans(n_clusters=k, n_init='auto',
                        random_state=seed).fit(data)
        labels = kmeans.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))
        silhouette_visualizer(kmeans, np.asarray(
            data), colors='yellowbrick', show=False)
        plt.savefig(cwd + f'silhouette_visualization_{k}')
        plt.show()

    plt.plot(clusters_number, sil, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Shilouette score')
    plt.title('The Shilouette Method')
    plt.savefig(cwd + 'SilhouetteM')
    plt.show()


def plot_results_2d(data, k, seed):
    # transform the data
    pca = PCA(2)
    df = pca.fit_transform(data)
    df.shape

    # plotting original data
    plt.scatter(df[:, 0], df[:, 1])
    plt.savefig(cwd + 'Original_Data_2d')
    plt.show()

    # initialize the class object
    kmeans = KMeans(n_clusters=k, n_init='auto',
                    random_state=seed)

    # predict the labels of clusters.
    label = kmeans.fit_predict(df)

    # getting unique labels
    u_labels = np.unique(label)

    # getting the Centroids
    centroids = kmeans.cluster_centers_

    # setting k different colors in cycler for plot
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        "color", mpl.cm.rainbow(np.linspace(0, 1, k)))

    # plotting the results:
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=f'C_{i}')
    plt.scatter(centroids[:, 0], centroids[:, 1],
                s=80, color='black', label='Centroids')
    plt.legend()
    plt.savefig(cwd + f'Clustered_Data_2d_k{k}_seed{seed}')
    plt.show()


def plot_results_3d(data, k, seed):
    # transform the data
    pca = PCA(3)
    df = pca.fit_transform(data)
    df.shape

    # building 3d figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plotting unclustered data
    ax.scatter3D(df[:, 0], df[:, 1], df[:, 2], c=df[:, 2], cmap='Greens')
    plt.savefig(cwd + 'Original_Data_3d')
    plt.show()

    # initialize the class object
    kmeans = KMeans(n_clusters=k, n_init='auto',
                    random_state=seed)

    # predict the labels of clusters.
    label = kmeans.fit_predict(df)

    # getting unique labels
    u_labels = np.unique(label)

    # getting the Centroids
    centroids = kmeans.cluster_centers_

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    colors = plt.cm.rainbow(np.linspace(0, 1, k))

    # plotting the results:
    for i in u_labels:
        ax.scatter3D(df[label == i, 0], df[label == i, 1],
                     df[label == i, 2], color=colors[i], label=f'C_{i}')
    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                 s=50, color='black', label='Centroids')
    ax.legend()
    plt.savefig(cwd + f'Clustered_Data_3d_k{k}_seed{seed}')
    plt.show()


def infered_results(data, features, k, seed):
    kmeans = KMeans(n_clusters=k, n_init='auto',
                    random_state=seed).fit(data)

    avg_big_five = np.round(kmeans.cluster_centers_)
    f_clusters = {}

    for i in range(k):
        f_clusters[i] = []

    for i, label in enumerate(kmeans.labels_):
        f_clusters[label].append(features[i])

    avg_features_per_cluster = []
    for cluster in f_clusters.values():
        c = np.asarray(cluster)
        avg_feat = []
        for i in range(5):
            avg_feat.append(np.sum(c[:, i]) / len(c))
        avg_features_per_cluster.append(np.rint(avg_feat))

    for i in range(k):
        print(f'{avg_big_five[i]} : {avg_features_per_cluster[i]}')


k_means()
