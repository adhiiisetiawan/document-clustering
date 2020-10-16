# import matplotlib.pyplot as plt
# from sklearn.metrics import davies_bouldin_score
# import numpy as np
# import tfidf
# from sklearn.preprocessing import normalize
# import pandas as pd
#
# def test_som(data,alpha,beta,maxEpoch,n_cluster):
#     epoch = 0
#
#     np_data = np.array(data)
#     np_weight = np.around(np.random.uniform(low=0, high=1, size=(n_cluster, len(np_data[0]))), 3)
#
#     while epoch < maxEpoch:
#         for x in np_data:
#             euclidianDistance = [sum((w - x) ** 2) for w in np_weight]
#             minimum = np.argmin(euclidianDistance)
#             np_weight[minimum] += alpha * (x - np_weight[minimum])
#         alpha *= beta
#         epoch += 1
#
#     label = []
#     d = {}
#     for i in range(1, n_cluster+1):
#         d["variable{0}".format(i)] = []
#
#     test = []
#     for i in d.keys():
#         test.append(i)
#
#     classified = []
#     for i in range(len(test)):
#         test[i] = []
#         classified.append(test[i])
#
#     for xTest in range (len(data)):
#         testing = [sum((weightTesting - data[xTest]) ** 2) for weightTesting in np_weight]
#         for i in range(len(classified)):
#             minimal = min(testing)
#             if minimal == testing[i]:
#                 label.append(i)
#                 break
#     return label
# print("------------------------ Test Akurasi-----------------------------------------")
# # thesis = pd.read_csv('document/Judul Skripsi TIF.csv')
# # filteredNan = thesis[thesis['judul_skripsi'].notnull()]
# # thesisTitle = filteredNan['judul_skripsi']
# # document = [title for title in thesisTitle]
# document = [
#     "Layang-layang terbang diangkasa",
#     "Burung- burung terbang diangkasa",
#     "Banyak layang-Layang berbentuk burung",
#     "Burung-burung di angkasa pulang di sore hari",
#     "Burung terbang untuk pulang ke sarang",
# ]
#
# tfidf_doc = tfidf.tfIdfFunction(document)
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
# dataFrame_tfidf = pd.DataFrame(tfidf_doc)
#
# norm = normalize(dataFrame_tfidf.values)
#
# label = []
# for i in range(2,14):
#     #print(test_som(norm,0.6,0.3,1,n_cluster=i))
#     label.append(test_som(norm,0.6,0.3,1,n_cluster=i))
# # print(label)
# db = {}
# for i in range(len(label)):
#     db[i+2] = davies_bouldin_score(norm,label[i])
# # print(db)
#
# plt.figure(figsize=(10, 10))
# plt.plot(list(db.keys()), list(db.values()))
# plt.xlabel("Number Cluster")
# plt.ylabel("Accuracy Values")
# plt.show()