import time
import os
import sys
import pandas as pd
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from DecisionTree import DecisionTree
from KNN import KNN
from LogisticRegression import LogisticRegression
from SVM import SVM
from NaiveBayes import NaiveBayes

# Load dataset
dataset = pd.read_csv("C:\\Users\\visha\\OneDrive\\Desktop\\heart.csv")
X = dataset.drop('target', axis=1)
y = dataset["target"]
X = X.values
y = y.values

# Initialize lists to store metrics
accuracy_scores = []
time_complexities = []
space_complexities = []

# Random Forest
start_time_rf = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf_rf = RandomForest(n_trees=20)
clf_rf.fit(X_train, y_train)
predictions_rf = clf_rf.predict(X_test)
acc_rf = np.sum(predictions_rf == y_test) / len(y_test) * 100
end_time_rf = time.time()
time_complexity_rf = end_time_rf - start_time_rf
space_complexity_rf = sys.getsizeof(X_train) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_test)

accuracy_scores.append(acc_rf)
time_complexities.append(time_complexity_rf)
space_complexities.append(space_complexity_rf)

# Decision Tree
start_time_dt = time.time()
clf_dt = DecisionTree(max_depth=10)
clf_dt.fit(X_train, y_train)
predictions_dt = clf_dt.predict(X_test)
acc_dt = np.sum(predictions_dt == y_test) / len(y_test) * 100
end_time_dt = time.time()
time_complexity_dt = end_time_dt - start_time_dt
space_complexity_dt = sys.getsizeof(X_train) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_test)

accuracy_scores.append(acc_dt)
time_complexities.append(time_complexity_dt)
space_complexities.append(space_complexity_dt)

# KNN
start_time_knn = time.time()
clf_knn = KNN(k=5)
clf_knn.fit(X_train, y_train)
predictions_knn = clf_knn.predict(X_test)
acc_knn = np.sum(predictions_knn == y_test) / len(y_test) * 100
end_time_knn = time.time()
time_complexity_knn = end_time_knn - start_time_knn
space_complexity_knn = sys.getsizeof(X_train) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_test)

accuracy_scores.append(acc_knn)
time_complexities.append(time_complexity_knn)
space_complexities.append(space_complexity_knn)

# Logistic Regression
start_time_lr = time.time()
clf_lr = LogisticRegression(lr=0.01)
clf_lr.fit(X_train, y_train)
predictions_lr = clf_lr.predict(X_test)
acc_lr = np.sum(predictions_lr == y_test) / len(y_test) * 100
end_time_lr = time.time()
time_complexity_lr = end_time_lr - start_time_lr
space_complexity_lr = sys.getsizeof(X_train) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_test)

accuracy_scores.append(acc_lr)
time_complexities.append(time_complexity_lr)
space_complexities.append(space_complexity_lr)

# SVM
start_time_svm = time.time()
clf_svm = SVM()
clf_svm.fit(X_train, y_train)
predictions_svm = clf_svm.predict(X_test)
acc_svm = np.sum(predictions_svm == y_test) / len(y_test) * 100
end_time_svm = time.time()
time_complexity_svm = end_time_svm - start_time_svm
space_complexity_svm = sys.getsizeof(X_train) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_test)

accuracy_scores.append(acc_svm)
time_complexities.append(time_complexity_svm)
space_complexities.append(space_complexity_svm)

# Naive Bayes
start_time_nb = time.time()
clf_nb = NaiveBayes()
clf_nb.fit(X_train, y_train)
predictions_nb = clf_nb.predict(X_test)
acc_nb = np.sum(predictions_nb == y_test) / len(y_test) * 100
end_time_nb = time.time()
time_complexity_nb = end_time_nb - start_time_nb
space_complexity_nb = sys.getsizeof(X_train) + sys.getsizeof(y_train) + sys.getsizeof(y_test) + sys.getsizeof(X_test)

accuracy_scores.append(acc_nb)
time_complexities.append(time_complexity_nb)
space_complexities.append(space_complexity_nb)

# Plot the accuracy, time complexities, and space complexities
algorithms = ["Random Forest", "Decision Tree", "KNN", "Logistic Regression", "SVM", "Naive Bayes"]

# Accuracy Plot
plt.figure(figsize=(15, 6))

plt.bar(algorithms, accuracy_scores, color='blue')
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.show()

#Time Plot
plt.figure(figsize=(15, 6))
plt.bar(algorithms, time_complexities, color='green')
plt.xlabel("Algorithms")
plt.ylabel("Time Complexity (seconds)")
plt.title("Time Complexity Comparison")
plt.show()

#Space plot
plt.figure(figsize=(15, 6))
plt.bar(algorithms, space_complexities, color='orange')
plt.xlabel("Algorithms")
plt.ylabel("Space Complexity (bytes)")
plt.title("Space Complexity Comparison")


plt.show()
