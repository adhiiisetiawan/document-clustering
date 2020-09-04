import preprocessing, som, tfidf_library
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

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
        "Layang-layang terbang diangkasa",
        "Burung- burung terbang diangkasa",
        "Banyak layang-Layang berbentuk burung",
        "Burung-burung di angkasa pulang di sore hari",
        "Burung terbang untuk pulang ke sarang",
    ]
    
#    dataTest = "Mobil dan bus tidak melewati persawahan. Ukuran nasi sangatlah kecil. Masakan kesukaan saya sekarang adalah nasi goreng. Sawah berukuran kecil hanya bisa ditanami sedikit beras. Test oke satu dua tiga empat lima enam tujuh delapan"

    preprocessing_doc = []
    
    for d in document:
        preprocessing_doc.append(preprocessing.text_preprocessing(d))
    
    document_weighting = tfidf_library.tfIdfCalculation(preprocessing_doc)
    print(document_weighting)
    
#    print(preprocessing_doc)
#    print(document_weighting)
#    
#    preprocessing_dataTest = preprocessing.text_preprocessing(dataTest)
#    document_weighting_dataTest = tfidf_library.tfIdfCalculation([preprocessing_dataTest])
    
#    print(document_weighting_dataTest)
    
    # som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 1, 3)
# Visualisasi
#     document_weighting, target = make_blobs(n_samples=300, n_features=2, centers=3)
#     centroids = som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 100, 3)
#     colors = 'rgbcmyk'
#
#     for x, label in zip(document_weighting, target):
#         plt.plot(x[0], x[1], colors[label] + '.')
#
#     plt.plot(centroids[:, 0], centroids[:, 1], 'kx')
#     plt.show()