
from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

random_seed_value = 2001
np.random.seed(random_seed_value)

def getVector1():
    df1 = pd.read_csv('./data-2014/nlpcc2014-ECS.csv')
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
    f = open('./data-2014/nlpcc2014-CHI.txt', 'r', encoding="utf-8")
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
    f = open('./data-2014/nlpcc2014-ESM.txt', 'r', encoding="utf-8")
    vector3= list()
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

def svm(x_train, x_test, y_train, y_test, k):
    input = open('./svm-model-2014' +'/'+ k, 'rb')
    clf = pickle.load(input)
    input.close()
    result = list()
    for j in x_test:
        result.append(clf.predict(j.reshape(1, -1)))
    result_train = list()
    for j in x_train:
        result_train.append(clf.predict(j.reshape(1, -1)))
    prob_train = list()
    for j in x_train:
        prob_train.append(clf.predict_proba(j.reshape(1, -1))[0][0])
        prob_train.append(clf.predict_proba(j.reshape(1, -1))[0][1])
    prob = list()
    accuracy = clf.score(x_train, y_train)
    for j in x_test:
        prob.append(clf.predict_proba(j.reshape(1, -1))[0][0])
        prob.append(clf.predict_proba(j.reshape(1, -1))[0][1])
    return result, prob, accuracy, prob_train,result_train

def getData(dataVector1, dataVector2, dataVector3):
    data = np.hstack((dataVector1, dataVector2, dataVector3))
    datan = data[5146:]
    datapp = data[0:5146]
    datap_index = np.random.choice(datapp.shape[0], 4658)
    datap = datapp[datap_index]
    data = np.vstack((datap, datan))
    return data[:, 0:25], data[:, 25:1025], data[:, 1025:]

def retrain2(prob1_train,prob2_train,prob1, prob2, y_test,y_train):
    a1=int(len(prob1_train) / 2)
    a2 = int(len(prob1) / 2)
    prob_train = np.hstack((np.array(prob1_train).reshape(a1, 2), np.array(prob2_train).reshape(a1, 2)))
    prob=np.hstack((np.array(prob1).reshape(a2,2), np.array(prob2).reshape(a2, 2)))
    clf = LogisticRegression(random_state=42)
    clf.fit(prob_train, y_train)
    acc = clf.score(prob_train, y_train)
    acc_pre = clf.score(prob, y_test)
    prob_train2 = list()
    for j in prob_train:
        prob_train2.append(clf.predict_proba(j.reshape(1, -1))[0][0])
    prob2 = list()
    for j in prob:
        prob2.append(clf.predict_proba(j.reshape(1, -1))[0][0])
    result = clf.predict(prob)
    return prob2, prob_train2, acc, acc_pre, result

def retrain3(prob1_train, prob2_train, prob3_train, prob1, prob2, prob3, y_test, y_train):
    a1 = int(len(prob1_train) / 2)
    a2 = int(len(prob1) / 2)
    prob_train = np.hstack((np.array(prob1_train).reshape(a1, 2), np.array(prob3_train).reshape(a1, 2),np.array(prob2_train).reshape(a1, 2)))
    prob = np.hstack((np.array(prob1).reshape(a2, 2), np.array(prob2).reshape(a2, 2), np.array(prob3).reshape(a2, 2)))
    clf = LogisticRegression(random_state=42)
    clf.fit(prob_train, y_train)
    acc_pre = clf.score(prob, y_test)
    acc = clf.score(prob_train, y_train)
    prob_train2 = list()
    for j in prob_train:
        prob_train2.append(clf.predict_proba(j.reshape(1, -1))[0][0])
    prob2 = list()
    for j in prob:
        prob2.append(clf.predict_proba(j.reshape(1, -1))[0][0])
    result = clf.predict(prob)
    return prob2, prob_train2, acc, acc_pre, result

def getAccuracy5(prob1, prob2, prob3, prob4, prob5, y_test, accur1, accur2, accur3, accur4, accur5):
    label = [0] * len(prob1)
    accura1 = accur1 / (accur1 + accur2 + accur3 + accur4 + accur5)
    accura2 = accur2 / (accur1 + accur2 + accur3 + accur4 + accur5)
    accura3 = accur3 / (accur1 + accur2 + accur3 + accur4 + accur5)
    accura4 = accur4 / (accur1 + accur2 + accur3 + accur4 + accur5)
    accura5 = accur5 / (accur1 + accur2 + accur3 + accur4 + accur5)
    probe = list()
    for i in range(0, len(prob1)):
        if ((prob1[i] * accura1 + prob2[i] * accura2 + prob3[i] * accura3 + prob4[i] * accura4 + prob5[i] * accura5) > (
                                    (1 - prob1[i]) * accura1 + (1 - prob2[i]) * accura2 + (1 - prob3[i]) * accura3 + (
                        1 - prob4[i]) * accura4 + (1 - prob5[i]) * accura5)):
            label[i] = 0
        elif (
            (prob1[i] * accura1 + prob2[i] * accura2 + prob3[i] * accura3 + prob4[i] * accura4 + prob5[i] * accura5) < (
                                    (1 - prob1[i]) * accura1 + (1 - prob2[i]) * accura2 + (1 - prob3[i]) * accura3 + (
                        1 - prob4[i]) * accura4 + (1 - prob5[i]) * accura5)):
            label[i] = 1
        else:
            label[i] = label[i] = np.random.randint(0, 1, 1)
        probe.append(
            prob1[i] * accura1 + prob2[i] * accura2 + prob3[i] * accura3 + prob4[i] * accura4 + prob5[i] * accura5)
    acc = 0
    for i in range(0, len(label)):
        if (label[i] == y_test[i]):
            acc = acc + 1
    return label, np.array(probe), acc / len(label)

dataVector1 = getVector1()
dataVector2 = getVector2()
dataVector3 = getVector3()
dataVector1, dataVector2, dataVector3 = getData(dataVector1, dataVector2, dataVector3)

y = np.concatenate((np.zeros(4658), np.ones(4658)))
skf = StratifiedKFold(n_splits=10, shuffle=True)

i = 0
accuracy = list()
pre_pos = list()
pre_neg = list()
rec_pos = list()
rec_neg = list()
f1_pos = list()
f1_neg = list()

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

    result1, prob1, accuracy1, prob1_train, result1_train = svm(x_train1, x_test1, y_train, y_test, 'svm_model' + str(i) + '-1.pkl')
    result2, prob2, accuracy2, prob2_train, result2_train = svm(x_train2, x_test2, y_train, y_test, 'svm_model' + str(i) + '-2.pkl')
    result3, prob3, accuracy3, prob3_train, result3_train = svm(x_train3, x_test3, y_train, y_test, 'svm_model' + str(i) + '-3.pkl')
    result4, prob4, accuracy4, prob4_train, result4_train = svm(x_train4, x_test4, y_train, y_test, 'svm_model' + str(i) + '-4.pkl')
    result5, prob5, accuracy5, prob5_train, result5_train = svm(x_train5, x_test5, y_train, y_test, 'svm_model' + str(i) + '-5.pkl')
    result6, prob6, accuracy6, prob6_train, result6_train = svm(x_train6, x_test6, y_train, y_test, 'svm_model' + str(i) + '-6.pkl')
    result7, prob7, accuracy7, prob7_train, result7_train = svm(x_train7, x_test7, y_train, y_test, 'svm_model' + str(i) + '-7.pkl')

    probe1, probe_train1, acc1, acc_pre1, result_pre1 = retrain2(prob1_train, prob6_train, prob1, prob6, y_test, y_train)
    probe2, probe_train2, acc2, acc_pre2, result_pre2 = retrain2(prob2_train, prob5_train, prob2, prob5, y_test, y_train)
    probe3, probe_train3, acc3, acc_pre3, result_pre3 = retrain2(prob3_train, prob4_train, prob3, prob4, y_test, y_train)
    probe4, probe_train4, acc4, acc_pre4, result_pre4 = retrain3(prob1_train, prob2_train, prob3_train, prob1, prob2, prob3, y_test, y_train)
    prob7 = np.array(prob7).reshape(int(len(prob7)/2), 2)[:, 0]

    label, probe, acc = getAccuracy5(probe1, probe2, probe3, probe4, prob7, y_test, acc1, acc2, acc3, acc4, accuracy7)

    print(i)
    print(classification_report(y_test, label, target_names=['0', '1'], digits=4))
    precision = metrics.precision_score(y_test, label, average=None)
    recall = metrics.recall_score(y_test, label, average=None)
    f1_score = metrics.f1_score(y_test, label, average=None)
    print(acc)
    accuracy.append(acc)
    pre_pos.append(precision[0])
    pre_neg.append(precision[1])
    rec_pos.append(recall[0])
    rec_neg.append(recall[1])
    f1_pos.append(f1_score[0])
    f1_neg.append(f1_score[1])


print("precision[pos]:", round(sum(pre_pos) / len(pre_pos), 4), "recall[pos]:", round(sum(rec_pos) / len(rec_pos), 4), "f1-score[pos]:", round(sum(f1_pos) / len(f1_pos), 4))
print("precision[neg]:", round(sum(pre_neg) / len(pre_neg), 4), "recall[neg]:", round(sum(rec_neg) / len(rec_neg), 4), "f1-score[neg]:", round(sum(f1_neg) / len(f1_neg), 4))
print("Accuracy:", round(sum(accuracy) / len(accuracy), 4))