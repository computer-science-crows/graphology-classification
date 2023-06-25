import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from Features.FeatureExtractor import FeaturesInfo
from Dataset.create_dataset import data 

# Load the handwriting images and extract the baseline, word slant, and word space features
# and save them in X_train and y_train
X_train = []
y_train = []
for i in range(1, 101):
    img = cv2.imread(f"handwriting{i}.jpg", 0)  # Load the grayscale image
   # baseline = extract_baseline(img)
   # slant = extract_word_slant(img)
   # space = extract_word_space(img)
   # features = [baseline, slant, space]
   # X_train.append(features)
   # y_train.append(get_personality_traits(f"handwriting{i}.txt"))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the KNN model on the training set

# CAMBIAR POR K MAS COMUN EN LA BIBLIOGRAFIA
k = 5  # number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict the personality traits of the testing set using the KNN model
y_pred = knn.predict(X_test)

# Evaluate the performance of the KNN model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
