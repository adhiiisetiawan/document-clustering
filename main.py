import preprocessing, som, tfidf_library

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

#    document = [
#            "Layang-layang terbang diangkasa",
#            "Burung- burung terbang diangkasa",
#            "Banyak layang-Layang berbentuk burung",
#            "Burung-burung di angkasa pulang di sore hari",
#            "Burung terbang untuk pulang ke sarang"
#    ]
    
    document = [
            "Sekarang saya sedang suka memasak. Masakan kesukaan saya sekarang adalah nasi goreng. Cara memasak nasi goreng adalah nasi digoreng",
            "Ukuran nasi sangatlah kecil, namun saya selalu makan nasi",
            "Nasi berasal dari beras yang ditanam di sawah. Sawah berukuran kecil hanya bisa ditanami sedikit beras",
            "Mobil dan bus dapat mengangkut banyak penumpang. Namun, bus berukuran jauh lebih besar dari mobil, apalagi mobil-mobilan",
            "Bus pada umumnya berukuran besar dan berpenumpang banyak, sehingga bus tidak bisa melewati persawahan"
    ]
    
#    dataTest = "Mobil dan bus tidak melewati persawahan. Ukuran nasi sangatlah kecil. Masakan kesukaan saya sekarang adalah nasi goreng. Sawah berukuran kecil hanya bisa ditanami sedikit beras. Test oke satu dua tiga empat lima enam tujuh delapan"

    preprocessing_doc = []
    
    for d in document:
        preprocessing_doc.append(preprocessing.text_preprocessing(d))
    
    document_weighting = tfidf_library.tfIdfCalculation(preprocessing_doc)
    # print(document_weighting)
    
#    print(preprocessing_doc)
#    print(document_weighting)
#    
#    preprocessing_dataTest = preprocessing.text_preprocessing(dataTest)
#    document_weighting_dataTest = tfidf_library.tfIdfCalculation([preprocessing_dataTest])
    
#    print(document_weighting_dataTest)
    
    # som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 1, 3)

# Visualisasi
    X, target = make_blobs(n_samples=30, n_features=2, centers=3)
    centroids = som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 100, 3)
    colors = 'rgbcmyk'

    for x, label in zip(X, target):
        plt.plot(x[0], x[1], colors[label] + '.')

    plt.plot(centroids[:, 0], centroids[:, 1], 'kx')
    plt.show()