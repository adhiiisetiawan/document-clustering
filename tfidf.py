import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfIdfCalculation(dataset):
    #mendeklarasikan object TfidfVectorizer
    tfIdfVectorizer = TfidfVectorizer(use_idf=True)

    #melakukan perhitungan tf idf dengan text yang diambil dari argument yg bernama "dataset"
    tfIdf = tfIdfVectorizer.fit_transform(dataset)

    #melakukan konversi hasil perhitungan tf idf ke dataframe
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])

    #sorting dataframe berdasarkan nilai tf-idf yang terbesar
    df = df.sort_values('TF-IDF', ascending=False)
    return df


def tfIdfCalculation_manual(dataset):
    # Remove duplicates value
    string = " "
    data = string.join(dataset)

    tokens = list(dict.fromkeys(data.split()))

    doc_fr = []
    for d in dataset:
        tmp = []
        for t in tokens:
            tmp.append([t, d.split().count(t)])
        doc_fr.append(tmp)

    # c = 1
    # for tf in doc_fr:
    #     print("doc", c)
    #     c += 1
    #     i = 0
    #     for h in tf:
    #         # if(h[1] != 0):
    #         #     tmp = list(h)
    #         #     tmp[1] = np.log10(h[1])
    #         #     tf[i] = tmp
    #         print(tf[i][0], "\t\t= ", tf[i][1])
    #         i += 1
    #     print("=========================")
    # print(doc_fr)
    #    # Create the pandas DataFrame
    #    df = pd.DataFrame(doc_fr, columns = ["doc 1", "doc 2", "doc 3"])
    #
    #    # print dataframe.
    #    print(df)
    return doc_fr