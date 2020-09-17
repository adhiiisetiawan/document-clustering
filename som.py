import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

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

    judul = document
    for xTest in range (len(data)):
        testing = [sum((weightTesting - data[xTest]) ** 2) for weightTesting in np_weight]
        # if testing[0] <= testing[1] and testing[0] <= testing[2]:
        if min(testing) == testing[0]:
            cluster1.append(judul[xTest])
        # elif testing[1] <= testing[0] and testing[1] <= testing[2]:
        elif min(testing) == testing[1]:
            cluster2.append(judul[xTest])
        # elif testing[2] <= testing[0] and testing[2] <= testing[1]:
        elif min(testing) == testing[2]:
            cluster3.append(judul[xTest])
        elif min(testing) == testing[3]:
            cluster4.append(judul[xTest])
        elif min(testing) == testing[4]:
            cluster5.append(judul[xTest])
        elif min(testing) == testing[5]:
            cluster6.append(judul[xTest])
    print("--------------------- Berapa Banyak Cluster -----------------")
    print("Cluster 1: ", len(cluster1))
    print("Cluster 2: ", len(cluster2))
    print("Cluster 3: ", len(cluster3))
    print("Cluster 4: ", len(cluster4))
    print("Cluster 5: ", len(cluster5))
    print("Cluster 6: ", len(cluster6))
    #
    # print("")
    print("---------------------Hasil cluster --------------------------")
    print("cluster 1: ", list(cluster1))
    print("cluster 2: ", list(cluster2))
    print("cluster 3: ", list(cluster3))
    print("cluster 4: ", list(cluster4))
    print("cluster 5: ", list(cluster5))
    print("cluster 6: ", list(cluster6))
    # print("")

    # wordcloud = WordCloud().generate(cluster1)
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()

    return " "