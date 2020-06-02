import sys
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


data = pd.read_csv("creditcard.csv")

data.shape

data["Class"].value_counts()

sample_data = data[:100000]
print(sample_data.shape)
print(sample_data["Class"].value_counts())

Fraud = sample_data[sample_data['Class']==1]
Valid = sample_data[sample_data['Class']==0]

print('Fraud Transaction: {}'.format(len(Fraud)))
print('Genuine Transaction: {}'.format(len(Valid)))

# data20000 = data_20000.drop(['Class'], axis=1)
# data20000.shape

# data20000_labels = data_20000["Class"]
# data20000_labels.shape



# corremat = sample_data.corr()
# fig = plt.figure(figsize = (10,8))

# sns.heatmap(corremat, vmax = 0.8, square = True)
# plt.show()


columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]


X = sample_data[columns]
Y = sample_data["Class"]
print(X.shape)
print(Y.shape)

X_Std = StandardScaler().fit_transform(X)
print(X_Std.shape)
print(type(X_Std))


X_train = X_Std[0:80000]
X_test = X_Std[80000:100000]
Y_train = Y[0:80000]
Y_test = Y[80000:100000]
#taking last 4k points as test data and first 16k points as train data

# myList = list(range(0,50))
# neighbors = list(filter(lambda x: x%2!=0, myList))  #This will give a list of odd numbers only ranging from 0 to 50

CV_Scores = []


KNN = KNeighborsClassifier(n_neighbors = 11, algorithm = 'kd_tree')
scores = model_selection.cross_val_score(KNN, X_train, Y_train, cv = 5, scoring='recall')
CV_Scores.append(scores.mean())


print(CV_Scores)

# k_value = neighbors[CV_Scores.index(max(CV_Scores))]

# knn_model = KNeighborsClassifier(n_neighbors = 11, algorithm = 'kd_tree')
# knn_model.fit(X_train, Y_train)
# prediction = knn_model.predict(X_test)
# # recall = recall_score(Y_test, prediction)
# # print('Recall Score of the knn classifier: {}'.format(recall))

# average_precision = average_precision_score(Y_test, prediction)
# print('Average precision-recall score: {0:0.2f}'.format(average_precision))

# disp = plot_precision_recall_curve(knn_model, X_test, Y_test)
# disp.ax_.set_title('2-class Precision-Recall curve:''AP={0:0.2f}'.format(average_precision))


# cm = confusion_matrix(Y_test, prediction)
# print(cm)
# print(classification_report(Y_test, prediction))

# k_range = range(1,15)
# scores = {}
# scores_list = []

# for k in k_range:
# 	knn_model = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
# 	knn_model.fit(X_train, Y_train)
# 	prediction = knn_model.predict(X_test)
# 	scores[k] = recall_score(Y_test, prediction)
# 	scores_list.append(recall_score(Y_test, prediction))


# plt.plot(k_range, scores_list)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Testing of recall')
# plt.show()










