import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data

data_frame_train =  np.load('/home/sheyla/university/4to/ML/graphology-classification/train.npy', allow_pickle=True)
big_five_train = np.transpose([i.big_five for i in data_frame_train])
X_train = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_frame_train]

data_frame_test =  np.load('/home/sheyla/university/4to/ML/graphology-classification/test.npy', allow_pickle=True)
big_five_test = np.transpose([i.big_five for i in data_frame_test])
X_test = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_frame_test]


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

big_five_train = group_2(big_five_train)
big_five_test = group_2(big_five_test)
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

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the KNN model on the training set

    # CAMBIAR POR K MAS COMUN EN LA BIBLIOGRAFIA
    k = 7  # number of neighbors to consider
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the personality traits of the testing set using the KNN model
    y_pred = knn.predict(X_test)

    # Evaluate the performance of the KNN model using accuracy
    return accuracy_score(y_test, y_pred)


Acurracy1 = KNN(X_train, X_test, y_train_neuroticism, y_test_neuroticism)
print("Accuracy : " + str(Acurracy1) + ' BigFive: Neuroticism   \n' )
Acurracy2 = KNN(X_train, X_test, y_train_extroversion, y_test_extroversion)
print("Accuracy: " + str(Acurracy2) + ' BigFive: Extroversion   \n' )
Acurracy3 = KNN(X_train, X_test, y_train_experienceOpenness, y_test_experienceOpenness)
print("Accuracy: " + str(Acurracy3) + ' BigFive: Experience Opennes  \n'   )
Acurracy4 = KNN(X_train, X_test, y_train_sympathy, y_test_sympathy)
print("Accuracy: " + str(Acurracy4) + ' BigFive: Simpaty   \n'  )
Acurracy5 = KNN(X_train, X_test, y_train_morality, y_test_morality)
print("Accuracy: " + str(Acurracy5) + ' BigFive: Morality   \n' )

