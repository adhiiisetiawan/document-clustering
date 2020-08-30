import numpy as np

# alpha = 0.6
# beta = 0.5
# maxEpoch = 100

def selfOrganizingMaps(data, alpha, beta, maxEpoch, n_cluster):


    np_data = np.array(data)
    np_weight = np.around(np.random.uniform(low=0, high=1, size=(n_cluster, len(data[0]))), 3)
    print("-------bobot pertamakali-------")
    print(np_weight)
    epoch = 0
    while epoch < maxEpoch:
        print("------------------------- Epoch ke", epoch + 1, "--------------------------------")
        for x in np_data:
            euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
            minimum = np.argmin(euclidianDistance)
            # update bobot
            np_weight[minimum] += alpha * (x - np_weight[minimum])
            print("--bobot setelah diupdate----")
            print(np_weight)
        alpha *= beta
        epoch += 1
    print("---bobot akhir---")
    return np_weight