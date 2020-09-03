import numpy as np

# alpha = 0.6
# beta = 0.5
# maxEpoch = 100

def selfOrganizingMaps(data, alpha, beta, maxEpoch):
    epoch = 0

    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(len(np_data), len(np_data[0]))), 3)
    print("ini bobot")
    print(np_weight)

    while epoch < maxEpoch:
        print("------------------------- Epoch ke", epoch + 1, "--------------------------------")
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            print("")
            print("============= Euclidian Distance ================")
            print("")
            print(euclidianDistance)
            print("")
            print("============= Jarak terdekat dari Euclidian Distance ==============")
            print("")
            minimum = np.argmin(euclidianDistance)
            print("pemenang = ", euclidianDistance[minimum])
            print("")
            print("=========== Update Bobot ============")
            np_weight[minimum] += alpha * (x - np_weight[minimum])
            print("")
            print(np_weight)
        alpha *= beta
        epoch += 1

    print("")
    print("============================================================================================================")
    print("Hasil Akhir Update Bobot")
    print("============================================================================================================")
    print(np_weight)
    print(" ")
    print("------------------ Testing ------------------")
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    # dataTest = [[0.1, 0.2, 0.3, 0.4],
    #            [0.5, 0.6, 0.7, 0.8],
    #            [0.9, 0.10, 0.11, 0.12]]
    for xTest in data:
        testing = [sum((weightTesting - xTest) ** 2) for weightTesting in np_weight]
        print(testing)
        print("")
        print("--------------- Pemenang --------------")
        if testing[0] <= testing[1] and testing[0] <= testing[2] and testing[0] <= testing[3]:
            print("pemenang =", testing[0], "termasuk cluster 1")
            cluster1.append(testing[0])
        elif testing[1] <= testing[0] and testing[1] <= testing[2] and testing[1] <= testing[3]:
            print("pemenang =", testing[1], "termasuk cluster 2")
            cluster2.append(testing[1])
        elif testing[2] <= testing[0] and testing[2] <= testing[1] and testing[2] <= testing[3]:
            print("pemenang =", testing[2], "termasuk cluster 3")
            cluster3.append(testing[2])
        elif testing[3] <= testing[0] and testing[3] <= testing[1] and testing[3] <= testing[2]:
            print("pemenang =", testing[3], "termasuk cluster 4")
            cluster4.append(testing[3])
        print("===============================")
        print("")
        print("---------------------Hasil cluster --------------------------")
        print("cluster 1: ", cluster1)
        print("cluster 2: ", cluster2)
        print("cluster 3: ", cluster3)
        print("cluster 4, ", cluster4)
    # c1 = np.array(cluster1)
    # c2 = np.array(cluster2)
    # c3 = np.array(cluster3)
    # print("--------------------------------final00000000000000")
