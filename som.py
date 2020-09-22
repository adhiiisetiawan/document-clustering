import numpy as np
import preprocessing
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import davies_bouldin_score

def selfOrganizingMaps(data, alpha, beta, maxEpoch,document):
    epoch = 0

    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(6, len(np_data[0]))), 3)

    while epoch < maxEpoch:
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            minimum = np.argmin(euclidianDistance)
            np_weight[minimum] += alpha * (x - np_weight[minimum])
        alpha *= beta
        epoch += 1

    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []
    label = []

    judul = document
    for xTest in range (len(data)):
        testing = [sum((weightTesting - data[xTest]) ** 2) for weightTesting in np_weight]
        if min(testing) == testing[0]:
            cluster1.append(judul[xTest])
            label.append(0)
        elif min(testing) == testing[1]:
            cluster2.append(judul[xTest])
            label.append(1)
        elif min(testing) == testing[2]:
            cluster3.append(judul[xTest])
            label.append(2)
        elif min(testing) == testing[3]:
            cluster4.append(judul[xTest])
            label.append(3)
        elif min(testing) == testing[4]:
            cluster5.append(judul[xTest])
            label.append(4)
        elif min(testing) == testing[5]:
            cluster6.append(judul[xTest])
            label.append(5)
    print("--------------------- Berapa Banyak Cluster -----------------")
    print("Cluster 1: ", len(cluster1))
    print("Cluster 2: ", len(cluster2))
    print("Cluster 3: ", len(cluster3))
    print("Cluster 4: ", len(cluster4))
    print("Cluster 5: ", len(cluster5))
    print("Cluster 6: ", len(cluster6))

    listCluster1 = np.array(cluster1).tolist()
    listCluster2 = np.array(cluster2).tolist()
    listCluster3 = np.array(cluster3).tolist()
    listCluster4 = np.array(cluster4).tolist()
    listCluster5 = np.array(cluster5).tolist()
    listCluster6 = np.array(cluster6).tolist()

    print("---------------------Hasil cluster --------------------------")
    print("cluster 1: ", listCluster1)
    print("cluster 2: ", listCluster2)
    print("cluster 3: ", listCluster3)
    print("cluster 4: ", listCluster4)
    print("cluster 5: ", listCluster5)
    print("cluster 6: ", listCluster6)
    # print("")

    print("")
    print("--------------Akurasi------------------")
    print("Hasil Akurasi: ", davies_bouldin_score(data, label))

    result1 = preprocessing.preprocessingWordCloud(str(listCluster1))
    result2 = preprocessing.preprocessingWordCloud(str(listCluster2))
    result3 = preprocessing.preprocessingWordCloud(str(listCluster3))
    result4 = preprocessing.preprocessingWordCloud(str(listCluster4))
    result5 = preprocessing.preprocessingWordCloud(str(listCluster5))
    result6 = preprocessing.preprocessingWordCloud(str(listCluster6))

    if cluster1:
        c1 = WordCloud().generate(result1)
        plt.imshow(c1, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster2:
        c2 = WordCloud().generate(result2)
        plt.imshow(c2, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster3:
        c3 = WordCloud().generate(result3)
        plt.imshow(c3, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster4:
        c4 = WordCloud().generate(result4)
        plt.imshow(c4, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster5:
        c5 = WordCloud().generate(result5)
        plt.imshow(c5, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster6:
        c6 = WordCloud().generate(result6)
        plt.imshow(c6, interpolation='bilinear')
        plt.axis("off")
        plt.show()


    return " "