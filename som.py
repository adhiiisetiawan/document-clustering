import numpy as np
import accuracy
import preprocessing
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import davies_bouldin_score

def selfOrganizingMaps(data, alpha, beta, maxEpoch,document, n_cluster):
    epoch = 0

    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(n_cluster, len(np_data[0]))), 3)

    while epoch < maxEpoch:
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            minimum = np.argmin(euclidianDistance)
            np_weight[minimum] += alpha * (x - np_weight[minimum])
        alpha *= beta
        epoch += 1

    label = []
    d = {}
    for i in range(1, n_cluster+1):
        d["variable{0}".format(i)] = []

    test = []
    for i in d.keys():
        test.append(i)

    classified = []
    for i in range(len(test)):
        test[i] = []
        classified.append(test[i])

    judul = document
    for xTest in range (len(data)):
        testing = [sum((weightTesting - data[xTest]) ** 2) for weightTesting in np_weight]
        for i in range(len(classified)):
            minimal = min(testing)
            if minimal == testing[i]:
                classified[i].append(judul[xTest])
                label.append(i)
                break
    print("----------Hasil Cluster-------------")
    for i in range(len(classified)):
        print("cluster {0} : ".format(i+1), classified[i])

    # print("----------label-----------------")
    # print("Label: ", label)

    print("----------Berapa Banyak Cluster-------------")
    for i in range(len(classified)):
        print("Banyaknya cluster {0} :".format(i+1), len(classified[i]))

    return " "

def pengujian(data,alpha,beta,maxEpoch,n_cluster):
    epoch = 0

    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(n_cluster, len(np_data[0]))), 3)

    while epoch < maxEpoch:
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            minimum = np.argmin(euclidianDistance)
            np_weight[minimum] += alpha * (x - np_weight[minimum])
        alpha *= beta
        epoch += 1

    label = []
    d = {}
    for i in range(1, n_cluster+1):
        d["variable{0}".format(i)] = []

    test = []
    for i in d.keys():
        test.append(i)

    classified = []
    for i in range(len(test)):
        test[i] = []
        classified.append(test[i])

    for xTest in range (len(data)):
        testing = [sum((weightTesting - data[xTest]) ** 2) for weightTesting in np_weight]
        for i in range(len(classified)):
            minimal = min(testing)
            if minimal == testing[i]:
                label.append(i)
                break
    return label