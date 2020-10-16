import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    thesis = pd.read_csv('document/Judul Skripsi TIF.csv')
    filteredNan = thesis[thesis['judul_skripsi'].notnull()]
    thesisTitle = filteredNan['judul_skripsi']
    document = [title for title in thesisTitle]

    import tfidf
    tfidf_doc = tfidf.tfIdfFunction(document)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dataFrame_tfidf = pd.DataFrame(tfidf_doc)

    norm = normalize(dataFrame_tfidf.values)
    import som
    som = som.selfOrganizingMaps(norm, 0.6, 0.5, 100, document, 6)
    print(som)


    label = []
    for i in range(2, 14):
        import som
        pengujian = som.pengujian(norm,0.6,0.3,1,n_cluster=i)
        label.append(pengujian)
    db = {}
    for i in range(len(label)):
        db[i + 2] = davies_bouldin_score(norm, label[i])
    # print(db)
    plt.figure(figsize=(10, 10))
    plt.plot(list(db.keys()), list(db.values()))
    plt.xlabel("Number Cluster")
    plt.ylabel("Accuracy Values")
    plt.show()
