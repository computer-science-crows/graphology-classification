import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data 

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


data_train =  np.load('C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/train.npy', allow_pickle=True)
data_test = np.load('C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/test.npy', allow_pickle=True)
data_validation = np.load('C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/validation.npy', allow_pickle=True)

y_train = np.transpose([i.big_five for i in data_train] + [i.big_five for i in data_test])
y_test = np.transpose([i.big_five for i in data_validation])
X_train = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_train] + [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_test]
X_test = [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_validation]

def ToDec(big_five):
    dec_values =  [0] * big_five.shape[1]
    for i in range(big_five.shape[1]):
        dec_values[i] = np.sum([big_five[j][i] * 2**(4 - j) for j in range(5)])
    return dec_values

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
    big_five[big_five <80] = 0
    big_five[big_five >= 80] = 1
 
    return big_five

# big_five = group_2(big_five)
# big_five = group_10(big_five)


y_train = group_10(y_train)
y_test  = group_10(y_test ) 

#Separar cada caracteristica a predecir
y_train_neuroticism = y_train[0]
y_train_extroversion = y_train[1]
y_train_experienceOpenness = y_train[2]
y_train_sympathy = y_train[3]
y_train_morality = y_train[4]

y_test_neuroticism = y_test[0]
y_test_extroversion = y_test[1]
y_test_experienceOpenness = y_test[2]
y_test_sympathy = y_test[3]
y_test_morality = y_test[4]

# y_train = np.array(ToDec(y_train))
# y_test = np.array(ToDec(y_test))


def SupportVectorMachine(X_train, X_test, y_train, y_test, kernel):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    accuracies = [0]*4 #Guardar accurracies para media
    precision = [0]*4 
    #Hacer 4-fold
    kf = KFold(n_splits= 4)
    splitted = kf.split(X_train)
    i = 0
    for train, validate in splitted:
        Xt, Xv, yt, yv = X_train[train], X_train[validate], y_train[train], y_train[validate]
        clf = SVC(decision_function_shape='ovo', kernel=kernel)
        clf.fit(Xt, yt)
        yp = clf.predict(Xv)
        accuracies[i] = accuracy_score(yv, yp)
        precision[i]= precision_score(yv, yp, average='macro', zero_division = 0)        
        i +=1
    

    #Crear SVM
    clf = SVC(decision_function_shape='ovo', kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return np.mean(accuracies), np.mean(precision), accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='macro', zero_division=0)


vector_linear1 = SupportVectorMachine(X_train, X_test, y_train_neuroticism, y_test_neuroticism, 'linear')
print("Accuracy : " + str(vector_linear1[2]) + '   Categories: 2   BigFive: Neuroticism   kernel: Lineal  \n' )
vector_linear2 = SupportVectorMachine(X_train, X_test, y_train_extroversion, y_test_extroversion, 'linear')
print("Accuracy: " + str(vector_linear2[2]) + '  Categories: 2   BigFive: Extroversion   kernel: Lineal  \n' )
vector_linear3 = SupportVectorMachine(X_train, X_test, y_train_experienceOpenness, y_test_experienceOpenness, 'linear')
print("Accuracy: " + str(vector_linear3[2]) + '  Categories: 2   BigFive: Experience Opennes  kernel: Lineal  \n'   )
vector_linear4 = SupportVectorMachine(X_train, X_test, y_train_sympathy, y_test_sympathy, 'linear')
print("Accuracy: " + str(vector_linear4[2]) + '  Categories: 2   BigFive: Simpaty  kernel: Lineal  \n'  )
vector_linear5 = SupportVectorMachine(X_train, X_test, y_train_morality, y_test_morality, 'linear')
print("Accuracy: " + str(vector_linear5[2]) + '  Categories: 2   BigFive: Morality  kernel: Lineal  \n' )
print("========================================================================================================")

vector_poly1 = SupportVectorMachine(X_train, X_test, y_train_neuroticism, y_test_neuroticism, 'poly')
print("Accuracy : " + str(vector_poly1[2]) + '   Categories: 2   BigFive: Neuroticism   kernel: Polinomial  \n' )
vector_poly2 = SupportVectorMachine(X_train, X_test, y_train_extroversion, y_test_extroversion, 'poly')
print("Accuracy: " + str(vector_poly2[2]) + '  Categories: 2   BigFive: Extroversion   kernel: Polinomial  \n' )
vector_poly3 = SupportVectorMachine(X_train, X_test, y_train_experienceOpenness, y_test_experienceOpenness, 'poly')
print("Accuracy: " + str(vector_poly3[2]) + '  Categories: 2   BigFive: Experience Opennes  kernel: Polinomial  \n'   )
vector_poly4 = SupportVectorMachine(X_train, X_test, y_train_sympathy, y_test_sympathy, 'poly')
print("Accuracy: " + str(vector_poly4[2]) + '  Categories: 2   BigFive: Simpaty  kernel: Polinomial  \n'  )
vector_poly5 = SupportVectorMachine(X_train, X_test, y_train_morality, y_test_morality, 'poly')
print("Accuracy: " + str(vector_poly5[2]) + '  Categories: 2   BigFive: Morality  kernel: Polinomial  \n' )
print("========================================================================================================")

vector_rbf1 = SupportVectorMachine(X_train, X_test, y_train_neuroticism, y_test_neuroticism, 'rbf')
print("Accuracy : " + str(vector_rbf1[2]) + '   Categories: 2   BigFive: Neuroticism   kernel: RBF  \n' )
vector_rbf2 = SupportVectorMachine(X_train, X_test, y_train_extroversion, y_test_extroversion, 'rbf')
print("Accuracy: " + str(vector_rbf2[2]) + '  Categories: 2   BigFive: Extroversion   kernel: RBF  \n' )
vector_rbf3 = SupportVectorMachine(X_train, X_test, y_train_experienceOpenness, y_test_experienceOpenness, 'rbf')
print("Accuracy: " + str(vector_rbf3[2]) + '  Categories: 2   BigFive: Experience Opennes  kernel: RBF  \n'   )
vector_rbf4 = SupportVectorMachine(X_train, X_test, y_train_sympathy, y_test_sympathy, 'rbf')
print("Accuracy: " + str(vector_rbf4[2]) + '  Categories: 2   BigFive: Simpaty  kernel: RBF  \n'  )
vector_rbf5 = SupportVectorMachine(X_train, X_test, y_train_morality, y_test_morality, 'rbf')
print("Accuracy: " + str(vector_rbf5[2]) + '  Categories: 2   BigFive: Morality  kernel: RBF  \n' )
print("========================================================================================================")

# bin_todec_linear = SupportVectorMachine(X_train, X_test, y_train, y_test, 'linear')
# print("Accuracy: " + str(bin_todec_linear[2]) + '   kernel: Lineal  \n'   )
# bin_todec_poly = SupportVectorMachine(X_train, X_test, y_train, y_test, 'poly')
# print("Accuracy: " + str(bin_todec_poly[2]) + '  kernel: Polinomial  \n'   )
# bin_todec_rbf = SupportVectorMachine(X_train, X_test, y_train, y_test, 'rbf')
# print("Accuracy: " + str(bin_todec_rbf[2]) + '  kernel: RBF  \n'   )



def GraphBarMean(accuracy, precision, title=""):
    labels = ["Neu","Extr", "Open", "Simp", "Mor"]
    fig, ax = plt.subplots()
    plt.title("Promedios de medidas en SVM (train/validation 5-Fold) "  + title)
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
    plt.title("Medidas para SVM (test) "  + title)
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
    plt.title("Medidas para SVM (test) " + title)
    plt.ylabel("Porcentaje del modelo")
    plt.xlabel("Kernels utilizados")
    val = np.arange(3)
    a =ax.bar(x = val - 0.2, height = accuracy, color = '#005aa0', width = 0.4 )
    p =ax.bar(x = val + 0.2, height = precision,  color = '#00ffe8', width = 0.4 )
    plt.xticks(val,labels)
    plt.legend((a,p), ("Accuracy", "Precision"))
    plt.show()  

def GraphBarREAL(accuracy, precision, title = ""):
    labels = ["Linear","Poly","RBF"]
    fig, ax = plt.subplots()
    plt.title("Medidas para SVM (test) " + title)
    plt.ylabel("Porcentaje del modelo")
    plt.xlabel("Kernels utilizados")
    val = np.arange(3)
    a =ax.bar(x = val - 0.2, height = accuracy, color = '#c40000', width = 0.4 )
    p =ax.bar(x = val + 0.2, height = precision,  color = '#ff3a3a', width = 0.4 )
    plt.xticks(val,labels)
    plt.legend((a,p), ("Accuracy", "Precision"))
    plt.show()    

def GraphPolygon(accuracy, precision, title = ""):
    labels = ["Linear","Poly","RBF"]
    plt.title("Medidas para SVM (test)" + title)
    plt.ylabel("Porcentaje del modelo")
    plt.xlabel("Kernels utilizados")
    val = np.arange(3)
    plt.plot(val, accuracy, color = "#ff6800", label = "Accuracy", linewidth= 3)
    plt.plot(val, precision, color = "#3cc314", label= "Precision", linewidth= 3)
    plt.plot(val, accuracy, 'o', color = "#ff6800")
    plt.plot(val, precision, 'o', color = "#3cc314")
    plt.xticks(val,labels)
    plt.legend()
    plt.show()  

#? Graficas para el train/validation 5 Fold

x_acurr_linear =  [vector_linear1[0], vector_linear2[0], vector_linear3[0], vector_linear4[0], vector_linear5[0]]
x_prec_linear =  [vector_linear1[1], vector_linear2[1], vector_linear3[1], vector_linear4[1], vector_linear5[1]]
GraphBarMean(x_acurr_linear, x_prec_linear, 'Kernel lineal')

x_acurr_poly =  [vector_poly1[0], vector_poly2[0], vector_poly3[0], vector_poly4[0], vector_poly5[0]]
x_prec_poly =  [vector_poly1[1], vector_poly2[1], vector_poly3[1], vector_poly4[1], vector_poly5[1]]
GraphBarMean(x_acurr_poly, x_prec_poly, 'Kernel Polinomial')

x_acurr_rbf =  [vector_rbf1[0], vector_rbf2[0], vector_rbf3[0], vector_rbf4[0], vector_rbf5[0]]
x_prec_rbf =  [vector_rbf1[1], vector_rbf2[1], vector_rbf3[1], vector_rbf4[1], vector_rbf5[1]]
GraphBarMean(x_acurr_rbf, x_prec_rbf, 'Kernel rbf')

#? Graficas para el test definitivo

x_acurr_linear =  [vector_linear1[2], vector_linear2[2], vector_linear3[2], vector_linear4[2], vector_linear5[2]]
x_prec_linear =  [vector_linear1[3], vector_linear2[3], vector_linear3[3], vector_linear4[3], vector_linear5[3]]
GraphBar(x_acurr_linear, x_prec_linear, 'Kernel lineal')

x_acurr_poly =  [vector_poly3[2], vector_poly2[2], vector_poly3[2], vector_poly4[2], vector_poly5[2]]
x_prec_poly =  [vector_poly3[3], vector_poly2[3], vector_poly3[3], vector_poly4[3], vector_poly5[3]]
GraphBar(x_acurr_poly, x_prec_poly, 'Kernel Polinomial')

x_acurr_rbf =  [vector_rbf3[2], vector_rbf2[2], vector_rbf3[2], vector_rbf4[2], vector_rbf5[2]]
x_prec_rbf =  [vector_rbf3[3], vector_rbf2[3], vector_rbf3[3], vector_rbf4[3], vector_rbf5[3]]
GraphBar(x_acurr_rbf, x_prec_rbf, 'Kernel rbf')


# #? Graficas para los que son llevando de binario a decimal
# accuracy = [bin_todec_linear[0], bin_todec_poly[0], bin_todec_rbf[0]]
# precision = [bin_todec_linear[1], bin_todec_poly[1], bin_todec_rbf[1]]
# GraphBarMEAN(accuracy, precision)

# accuracy = [bin_todec_linear[2], bin_todec_poly[2], bin_todec_rbf[2]]
# precision = [bin_todec_linear[3], bin_todec_poly[3], bin_todec_rbf[3]]
# GraphBarREAL(accuracy, precision)
# GraphPolygon(accuracy, precision)