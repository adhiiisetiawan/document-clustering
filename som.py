import numpy as np
from wordcloud import WordCloud
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
    print("cluster 1: ", np.array(cluster1).tolist())
    print("cluster 2: ", np.array(cluster2).tolist())
    print("cluster 3: ", np.array(cluster3).tolist())
    print("cluster 4: ", np.array(cluster4).tolist())
    print("cluster 5: ", np.array(cluster5).tolist())
    print("cluster 6: ", np.array(cluster6).tolist())
    # print("")


    if cluster1:
        c1 = WordCloud().generate(str(np.array(cluster1).tolist()))
        plt.imshow(c1, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster2:
        c2 = WordCloud().generate(str(np.array(cluster2).tolist()))
        plt.imshow(c2, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster3:
        c3 = WordCloud().generate(str(np.array(cluster3).tolist()))
        plt.imshow(c3, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster4:
        c4 = WordCloud().generate(str(np.array(cluster4).tolist()))
        plt.imshow(c4, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster5:
        c5 = WordCloud().generate(str(np.array(cluster5).tolist()))
        plt.imshow(c5, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    if cluster6:
        c6 = WordCloud().generate(str(np.array(cluster6).tolist()))
        plt.imshow(c6, interpolation='bilinear')
        plt.axis("off")
        plt.show()


    return " "