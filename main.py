import preprocessing, som, tfidf
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    thesis = pd.read_csv('Judul Skripsi TIF.csv')
    thesis = thesis.drop(thesis.columns[[0, 1, 3]], axis=1)
    document = thesis.values

    tfidf_doc = tfidf.tfIdfFunction(document)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dataFrame_tfidf = pd.DataFrame(tfidf_doc)
    # print(test2)

    from sklearn.preprocessing import normalize

    norm = normalize(dataFrame_tfidf.values)

    som = som.selfOrganizingMaps(norm, 0.6, 0.5, 100, document)
    print(som)
