import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import som
import tfidf
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    thesis = pd.read_csv('document/Judul Skripsi TIF.csv')
    filteredNan = thesis[thesis['judul_skripsi'].notnull()]
    thesisTitle = filteredNan['judul_skripsi']
    document = [title for title in thesisTitle]

    tfidf_doc = tfidf.tfIdfFunction(document)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dataFrame_tfidf = pd.DataFrame(tfidf_doc)

    norm = normalize(dataFrame_tfidf.values)
    #Bagian ini
    # som_final = som.selfOrganizingMaps(norm, 0.6, 0.5, 100, document, 6)
    # print(som_final)


    label = []
    for i in range(2, 13):
        # dan ini
        # pengujian = som.pengujian(norm,0.6,0.5,1,n_cluster=i)
        print("Cluster ke-",i)
        som_final = som.selfOrganizingMaps(norm, 0.6, 0.5, 100, document, n_cluster=i)
        label.append(som_final)
    db = {}
    for i in range(len(label)):
        db[i + 2] = davies_bouldin_score(norm, label[i])
    # print(db)
    plt.figure(figsize=(10, 10))
    plt.plot(list(db.keys()), list(db.values()), marker='o', color='red', linewidth=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.title("Accuracy of Each Cluster", fontsize=25)
    plt.xlabel("Number Cluster", fontsize=20)
    plt.ylabel("Accuracy Values", fontsize=20)
    plt.savefig("img/Akurasi.png", bbox_inches='tight', dpi=100)
    plt.show()
