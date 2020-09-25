import matplotlib.pyplot as plt
import som
from sklearn.metrics import davies_bouldin_score

def accuracy(data, label, document):
    db = {}
    for i in range(2, 10):
        # selfOrganizingMaps = som.selfOrganizingMaps(data, 0.6, 0.5, 100, document, n_cluster=i)
        db[i] = davies_bouldin_score(data, label)

    plt.figure(figsize=(5, 5))
    plt.plot(list(db.keys()), list(db.values()))
    plt.xlabel("Number Cluster")
    plt.ylabel("Accuracy Values")
    plt.show()