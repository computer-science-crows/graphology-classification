import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data
import scipy.stats as stats

# load dataset
data_train =  np.load('C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/train.npy', allow_pickle=True)
data_test = np.load('C:/Users/User/Desktop/Ciber/MATCOM/Cuarto/ML/graphology-classification/test.npy', allow_pickle=True)

Y =  np.transpose([i.big_five for i in data_train] + [i.big_five for i in data_test])
X =  [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_train] + [[i.features.baseline, i.features.line_space, i.features.margin, i.features.slant, i.features.word_space] for i in data_test]

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


Y10 = group_10(Y)
Y = group_2(Y)

Y_neuroticism10 = Y10[0]
Y_extroversion10 = Y10[1]
Y_experienceOpenness10 = Y10[2]
Y_sympathy10 = Y10[3]
Y_morality10 = Y10[4]

Y_neuroticism = Y[0]
Y_extroversion = Y[1]
Y_experienceOpenness = Y[2]
Y_sympathy = Y[3]
Y_morality = Y[4]

# prepare models
models = []
models.append(('SVMpoly', SVC(decision_function_shape='ovo', kernel='poly')))
models.append(('SVMlinear', SVC(decision_function_shape='ovo', kernel='linear')))
models.append(('SVMrbf', SVC(decision_function_shape='ovo', kernel='rbf')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))


def evaluate_models(X,Y, models, Y10= None, Caracteristic = None):

    if Caracteristic: print(Caracteristic)
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_test = scaler.transform(X)
    
    # prepare configuration for cross validation test harness
    seed = 7
    
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(f'{name}_2')
        msg = "%s: Mean %f, Standard Deviation (%f)" % (f'{name}_2', cv_results.mean(), cv_results.std())
        print(msg)

    if Y10 is not None:
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
            cv_results = model_selection.cross_val_score(model, X, Y10, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(f'{name}_10')
            msg = "%s: Mean %f, Standard Deviation (%f)" % (f'{name}_10', cv_results.mean(), cv_results.std())
            print(msg)
    
    print("<===================================================>")

    for i in range(len(results)):
        for j in range(i+1,len(results)):
            statistic, p_value = stats.ttest_rel(results[i],results[j])
            msg = "%s <-> %s: statistic %f, p-value: %f" % (names[i], names[j], statistic, p_value)
            print(msg)

    print("+-----------------------------------------------------+")
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Comparación de algoritmos'+ f'({Caracteristic})')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


evaluate_models(X,Y_neuroticism,models,Y_neuroticism10,"Neuroticism")
evaluate_models(X,Y_extroversion,models,Y_extroversion10,"Extroversion")
evaluate_models(X,Y_experienceOpenness,models,Y_experienceOpenness10,"ExperienceOpenness")
evaluate_models(X,Y_sympathy,models,Y_sympathy10,"Sympathy")
evaluate_models(X,Y_morality,models,Y_morality10,"Morality")




