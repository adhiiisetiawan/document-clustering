import numpy as np

# alpha = 0.6
# beta = 0.5
# maxEpoch = 100

def selfOrganizingMaps(data, alpha, beta, maxEpoch, dataTest):
    epoch = 0

    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(3, 4)), 3)
    # print(np_data)
    # print(np_weight)

    while epoch < maxEpoch:
        print("------------------------- Epoch ke", epoch + 1, "--------------------------------")
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            print("============= Euclidian Distance ================")
            print("")
            print(euclidianDistance)
            print("")
            print("Euclidian Distance 1: ", euclidianDistance[0])
            print("Euclidian Distance 2: ", euclidianDistance[1])
            print("Euclidian Distance 3: ", euclidianDistance[2])

            print("")
            print("============= Jarak terdekat dari Euclidian Distance ==============")
            print("")
            minimum = np.argmin(euclidianDistance)
            print("pemenang = ", euclidianDistance[minimum])
            print("=========== Update Bobot ============")
            print("")
            np_weight[minimum] += alpha * (x - np_weight[minimum])
            print(np.around(np_weight, 2))
        alpha *= beta
        epoch += 1

    print("------------------ Testing ------------------")
    cluster1 = []
    cluster2 = []
    cluster3 = []

    datatesting = np.array(dataTest)

    for xTest in datatesting:
        testing = [sum((weightTesting - xTest) ** 2) for weightTesting in np_weight]
        print("Euclidian Distance 1:", testing[0])
        print("Euclidian Distance 2", testing[1])
        print("Euclidian Distance 3", testing[2])
        print("")
        print("--------------- Pemenang --------------")
        if testing[0] <= testing[1] and testing[0] <= testing[2]:
            print("pemenang =", testing[0], "termasuk cluster 1")
            cluster1.append(testing[0])
        elif testing[1] <= testing[0] and testing[1] <= testing[2]:
            print("pemenang =", testing[1], "termasuk cluster 2")
            cluster2.append(testing[1])
        elif testing[2] <= testing[0] and testing[2] <= testing[1]:
            print("pemenang =", testing[2], "termasuk cluster 3")
            cluster3.append(testing[2])
        print("===============================")
        print("")
        print("---------------------Hasil cluster --------------------------")
        print("cluster 1: ", cluster1)
        print("cluster 2: ", cluster2)
        print("cluster 3: ", cluster3)