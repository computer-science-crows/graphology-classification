import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

data_frame_train =  np.load('/home/sheyla/university/4to/ML/graphology-classification/train.npy', allow_pickle=True)
data_frame_test =  np.load('/home/sheyla/university/4to/ML/graphology-classification/test.npy', allow_pickle=True)

big_five_train_test = np.transpose([i.big_five for i in data_frame_train] + [i.big_five for i in data_frame_test])
X_train_test = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_frame_train] + [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_frame_test]

data_frame_val =  np.load('/home/sheyla/university/4to/ML/graphology-classification/validation.npy', allow_pickle=True)
big_five_val = np.transpose([i.big_five for i in data_frame_val])
X_val = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_frame_val]

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

def group_2(big_five):
    big_five = np.array(big_five)
    big_five[big_five < 80] = 0
    big_five[big_five >= 80] = 1
 
    return big_five

big_five_train = group_2(big_five_train_test)
big_five_test = group_2(big_five_val)
# big_five_train = group_10(big_five_train)
# big_five_test = group_10(big_five_test)

#Separar cada caracteristica a predecir
y_train_neuroticism = big_five_train[0]
y_train_extroversion = big_five_train[1]
y_train_experienceOpenness = big_five_train[2]
y_train_sympathy = big_five_train[3]
y_train_morality = big_five_train[4]

y_test_neuroticism = big_five_test[0]
y_test_extroversion = big_five_test[1]
y_test_experienceOpenness = big_five_test[2]
y_test_sympathy = big_five_test[3]
y_test_morality = big_five_test[4]


def KNN(X_train, X_test, y_train, y_test):

    k = 7  # number of neighbors to consider
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    accuracies = [0]*4 #Guardar accurracies para media
    precision = [0]*4 

    kf = KFold(n_splits= 4)
    splitted = kf.split(X_train)
    i = 0
    for train, validate in splitted:
        Xt, Xv, yt, yv = X_train[train], X_train[validate], y_train[train], y_train[validate]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Xt, yt)

        # Predict the personality traits of the testing set using the KNN model
        yp = knn.predict(Xv)
        accuracies[i] = accuracy_score(yv, yp)
        precision[i]= precision_score(yv, yp, average='macro', zero_division = 0)        
        i +=1

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the personality traits of validate set using the KNN model
    y_pred = knn.predict(X_test)

    # Evaluate the performance of the KNN model using accuracy
    return np.mean(accuracies), np.mean(precision), accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro', zero_division=0)


Acurracy1 = KNN(X_train_test, X_val, y_train_neuroticism, y_test_neuroticism)
print("Accuracy : " + str(Acurracy1) + ' BigFive: Neuroticism   \n' )
Acurracy2 = KNN(X_train_test, X_val, y_train_extroversion, y_test_extroversion)
print("Accuracy: " + str(Acurracy2) + ' BigFive: Extroversion   \n' )
Acurracy3 = KNN(X_train_test, X_val, y_train_experienceOpenness, y_test_experienceOpenness)
print("Accuracy: " + str(Acurracy3) + ' BigFive: Experience Opennes  \n'   )
Acurracy4 = KNN(X_train_test, X_val, y_train_sympathy, y_test_sympathy)
print("Accuracy: " + str(Acurracy4) + ' BigFive: Simpaty   \n'  )
Acurracy5 = KNN(X_train_test, X_val, y_train_morality, y_test_morality)
print("Accuracy: " + str(Acurracy5) + ' BigFive: Morality   \n' )


def GraphBarMean(accuracy, precision, title=""):
    labels = ["Neu","Extr", "Open", "Simp", "Mor"]
    fig, ax = plt.subplots()
    plt.title("Promedios de medidas en KNN (train/validation 5-Fold) "  + title)
    plt.ylabel("Porcentaje del modelo")
    plt.xlabel("Categorias del BigFive")
    val = np.arange(5)
    a =ax.bar(x = val - 0.2, height = accuracy, color = '#005aa0', width = 0.4 )
    p =ax.bar(x = val + 0.2, height = precision,  color = '#00ffe8', width = 0.4 )
    plt.xticks(val,labels)
    plt.legend((a,p), ("Accuracy", "Precision"))
    plt.show()

def GraphBar(accuracy, precision, title=""):
    labels = ["Neu","Extr", "Open", "Simp", "Mor"]
    fig, ax = plt.subplots()
    plt.title("Medidas para KNN (test) "  + title)
    plt.ylabel("Porcentaje del modelo")
    plt.xlabel("Categorias del BigFive")
    val = np.arange(5)
    a =ax.bar(x = val - 0.2, height = accuracy, color = '#c40000', width = 0.4 )
    p =ax.bar(x = val + 0.2, height = precision, color = '#ff3a3a', width = 0.4 )
    plt.xticks(val,labels)
    plt.legend((a,p), ("Accuracy", "Precision"))
    plt.show()

def GraphBarMEAN(accuracy, precision, title = ""):
    labels = ["Linear","Poly","RBF"]
    fig, ax = plt.subplots()
    plt.title("Medidas para KNN (test) " + title)
    plt.ylabel("Porcentaje del modelo")
    plt.xlabel("Kernels utilizados")
    val = np.arange(3)
    a =ax.bar(x = val - 0.2, height = accuracy, color = '#005aa0', width = 0.4 )
    p =ax.bar(x = val + 0.2, height = precision,  color = '#00ffe8', width = 0.4 )
    plt.xticks(val,labels)
    plt.legend((a,p), ("Accuracy", "Precision"))
    plt.show()  


# Graficas para el train/validation 5 Fold
x_acurr_linear =  [Acurracy1[0], Acurracy2[0], Acurracy3[0], Acurracy4[0], Acurracy5[0]]
x_prec_linear =  [Acurracy1[1], Acurracy2[1], Acurracy3[1], Acurracy4[1], Acurracy5[1]]
GraphBarMean(x_acurr_linear, x_prec_linear)

# Graficas para el test definitivo
x_acurr_linear =  [Acurracy1[2], Acurracy2[2], Acurracy3[2], Acurracy4[2], Acurracy5[2]]
x_prec_linear =  [Acurracy1[3], Acurracy2[3], Acurracy3[3], Acurracy4[3], Acurracy5[3]]
GraphBar(x_acurr_linear, x_prec_linear)
