import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data 

data_frame =  np.load('C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/Algorithms/dataset_feat.npy', allow_pickle=True)

big_five = np.transpose([i.big_five for i in data_frame])
X_value = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_frame]

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

big_five = group_10(big_five)

#Separar cada caracteristica a predecir
y_value_neuroticism = big_five[0]
y_value_extroversion = big_five[1]
y_value_experienceOpenness = big_five[2]
y_value_sympathy = big_five[3]
y_value_morality = big_five[4]

#Separa los conjuntode entrenamiento y test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_value, y_value_neuroticism, test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_value, y_value_extroversion, test_size=0.2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_value, y_value_experienceOpenness, test_size=0.2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_value, y_value_sympathy, test_size=0.2)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_value, y_value_morality, test_size=0.2)

def SupportVectorMachine(X_train, X_test, y_train, y_test, kernel):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Crear SVM
    clf = SVC(decision_function_shape='ovo', kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(y_test)
    return accuracy_score(y_test, y_pred)

accuracy1 = SupportVectorMachine(X_train1, X_test1, y_train1, y_test1, 'linear')
accuracy2 = SupportVectorMachine(X_train2, X_test2, y_train2, y_test2, 'linear')
accuracy3 = SupportVectorMachine(X_train3, X_test3, y_train3, y_test3, 'linear')
accuracy4 = SupportVectorMachine(X_train4, X_test4, y_train4, y_test4, 'linear')
accuracy5 = SupportVectorMachine(X_train5, X_test5, y_train5, y_test5, 'linear')
