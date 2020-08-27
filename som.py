import numpy as np

# alpha = 0.6
# beta = 0.5
# maxEpoch = 100

def selfOrganizingMaps(data, alpha, beta, maxEpoch):
    epoch = 0

    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(3, len(np_data))), 3)
    # print(np_data)
    print("--------Bobot Pertama kali-----")
    print(np_weight)

    # weight = [[0.2, 0.6, 0.5, 0.9],
    #           [0.8, 0.4, 0.7, 0.3]]
    #
    # np_weight = np.array(weight)

    while epoch < maxEpoch:
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            minimum = np.argmin(euclidianDistance)

            # Update bobot
            np_weight[minimum] += alpha * (x - np_weight[minimum])
            # print(np.around(np_weight, 2))
            print("---------setelah update bobot----")
            print(np_weight)
        alpha *= beta
        epoch += 1
    print("----hasil akhir bobot----")
    return np_weight