
from __future__ import division
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle

random_seed_value = 1338
np.random.seed(random_seed_value)

def getVector1():
    df1 = pd.read_csv('./data-2013/nlpcc2013-ECS.csv')
    data1 = list()
    for i in range(len(df1['target'])):
        dl = list()
        for j in df1.keys():
            if j != 'target':
                dl.append(df1[j][i])
        data1.append(dl)
    data1 = np.array(data1)
    return data1

def getVector2():
    f = open('./data-2013/nlpcc2013-CHI.txt', 'r', encoding="utf-8")
    vector2 = list()
    while True:
        line = f.readline()
        if line:
            line = line.strip("\n")
            line = line.split(",")
            vec = list()
            for i in line:
                if (i.strip()):
                    vec.append(float(i))
        else:
            break
        vector2.append(vec)
    return np.array(vector2)

def getVector3():
    f = open('./data-2013/nlpcc2013-ESM.txt', 'r', encoding="utf-8")
    vector3 = list()
    while True:
        line = f.readline()
        if line:
            line = line.strip("\n")
            line = line.split(",")
            vec = list()
            for i in line:
                if (i.strip()):
                    vec.append(float(i))
        else:
            break
        vector3.append(vec)
    return np.array(vector3)

def svmmodel(x_train, x_test, y_train, y_test, k):
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(x_train, y_train)
    output = open('./svm-model-2013' + '/svm_model' + k + '.pkl', 'wb')
    s = pickle.dump(clf, output)
    output.close()
    return s

def getData(dataVector1, dataVector2, dataVector3):
    data = np.hstack((dataVector1, dataVector2, dataVector3))
    datap = data[0:3639]
    datann = data[3639:-1]
    datan_index = np.random.choice(datann.shape[0], 3639)
    datan = datann[datan_index]
    data = np.vstack((datap, datan))
    return data[:, 0:25], data[:, 25:1025], data[:, 1025:]

dataVector1 = getVector1()
dataVector2 = getVector2()
dataVector3 = getVector3()
dataVector1, dataVector2, dataVector3 = getData(dataVector1, dataVector2, dataVector3)

i = 0
y = np.concatenate((np.zeros(3639), np.ones(3639)))
skf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in skf.split(dataVector1, y):
    x_train1 = dataVector1[train_index]
    x_test1 = dataVector1[test_index]
    x_train2 = dataVector2[train_index]
    x_test2 = dataVector2[test_index]
    x_train3 = dataVector3[train_index]
    x_test3 = dataVector3[test_index]
    x_train4 = np.hstack((x_train1, x_train2))
    x_test4 = np.hstack((x_test1, x_test2))
    x_train5 = np.hstack((x_train1, x_train3))
    x_test5 = np.hstack((x_test1, x_test3))
    x_train6 = np.hstack((x_train2, x_train3))
    x_test6 = np.hstack((x_test2, x_test3))
    x_train7 = np.hstack((x_train1, x_train2, x_train3))
    x_test7 = np.hstack((x_test1, x_test2, x_test3))

    y_train = y[train_index]
    y_test = y[test_index]
    i = i + 1

    svmmodel(x_train1, x_test1, y_train, y_test, str(i) + '-1')
    svmmodel(x_train2, x_test2, y_train, y_test, str(i) + '-2')
    svmmodel(x_train3, x_test3, y_train, y_test, str(i) + '-3')
    svmmodel(x_train4, x_test4, y_train, y_test, str(i) + '-4')
    svmmodel(x_train5, x_test5, y_train, y_test, str(i) + '-5')
    svmmodel(x_train6, x_test6, y_train, y_test, str(i) + '-6')
    svmmodel(x_train7, x_test7, y_train, y_test, str(i) + '-7')
