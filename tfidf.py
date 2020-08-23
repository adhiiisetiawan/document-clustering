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