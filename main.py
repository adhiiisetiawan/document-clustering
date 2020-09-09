import som, tfidf
import pandas as pd
from sklearn.preprocessing import normalize

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    thesis = pd.read_csv('document/Judul Skripsi TIF.csv')
    thesis = thesis.drop(thesis.columns[[0, 1, 3]], axis=1)
    document = thesis.values

    tfidf_doc = tfidf.tfIdfFunction(document)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dataFrame_tfidf = pd.DataFrame(tfidf_doc)

    norm = normalize(dataFrame_tfidf.values)

    som = som.selfOrganizingMaps(norm, 0.6, 0.5, 100, document)
    print(som)
