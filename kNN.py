# -*- coding:utf-8 -*-
import os
import numpy as np

import csv


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data_folder = os.getcwd()
data_filename = os.path.join(data_folder, "ionosphere.data")


X = np.zeros((351, 34), dtype='float')
#预处理
X_transformed = MinMaxScaler().fit_transform(X)

y = np.zeros((351,), dtype='bool')

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)

    #获取每个个体前34个值
    for i, row in enumerate(reader):
        data = [float(datum) for datum in row[:-1]]
        X[i] = data
        y[i] = row[-1] == 'g'

# 创建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)

#训练
# estimator.fit(X_train, y_train)
#测试
# y_predicted = estimator.predict(X_test)
# accuracy = np.mean(y_test == y_predicted) * 100
# print("The accuracy is {0:.1f}%".format(accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21)) #调参

for n_neighbors in parameter_values:
    #导入K近邻分类器
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)


plt.figure(figsize=(32,20))
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)


#流水线
from sklearn.pipeline import Pipeline
scaling_pipeline = Pipeline([('scale', MinMaxScaler()),('predict', KNeighborsClassifier())])
scores = cross_val_score(scaling_pipeline, X, y, scoring='accuracy')
