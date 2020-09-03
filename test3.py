# document = [
#     "Layang-layang terbang diangkasa",
#     "Burung- burung terbang diangkasa",
#     "Banyak layang-Layang berbentuk burung",
#     "Burung-burung di angkasa pulang di sore hari",
#     "Burung terbang untuk pulang ke sarang",
# ]
document = [
    "tani tindak agroindustri basis agraris padi",
    "tani sayur komoditas sayur kebun",
    "padi komoditas pokok produksi tanam padi",
    "tanam karet komoditas kebun ekspor hujan",
]
#document = [
#        "Sekarang saya sedang suka memasak. Masakan kesukaan saya sekarang adalah nasi goreng. Cara memasak nasi goreng adalah nasi digoreng",
#        "Ukuran nasi sangatlah kecil, namun saya selalu makan nasi",
#        "Nasi berasal dari beras yang ditanam di sawah. Sawah berukuran kecil hanya bisa ditanami sedikit beras",
#        "Mobil dan bus dapat mengangkut banyak penumpang. Namun, bus berukuran jauh lebih besar dari mobil, apalagi mobil-mobilan",
#        "Bus pada umumnya berukuran besar dan berpenumpang banyak, sehingga bus tidak bisa melewati persawahan"
#]
########################################################################################################################
#buat list.append untuk bikin list yg isinya setiap kalimat yg setelah dipreprocessing
preprocessing_doc = []
import preprocessing
for d in document:
    preprocessing_doc.append(preprocessing.text_preprocessing(d))
print("================================================================================================================")
print("Ini hasil Preprocessing Setiap Dokumen2")
print("================================================================================================================")
print(preprocessing_doc)
print("")

########################################################################################################################
#membuat daftar bag of words yang isinya kata2 displit setiap dokumen
list_bow = []
for d in range(len(preprocessing_doc)):
    bow = preprocessing_doc[d].split(" ")
    list_bow.append(bow)
print("================================================================================================================")
print("Ini Daftar Bag Of Words")
print("================================================================================================================")
print(list_bow)
print(" ")


########################################################################################################################
#untuk membuat set kata (kata unik)
preprocessing_doc = []
for d in document:
    preprocessing_doc.append(preprocessing.text_preprocessing(d))
gabungan = " ".join(preprocessing_doc)
kata2 = gabungan.split()
print("================================================================================================================")
print("Ini WordSet")
print("================================================================================================================")
wordset = set(kata2)
print(wordset)
print("")
########################################################################################################################

#membuat method pengitung kata2
def word_count(str):
    w = dict.fromkeys(wordset,0)
    words = str.split()

    for word in words:
        if word in w:
            w[word] += 1
        else:
            w[word] = 1

    return w
#append dictonary ke list
list = []
for kalimat in preprocessing_doc:
    hasil = word_count(kalimat)
    list.append(hasil)
print("================================================================================================================")
print("List of Dictonary")
print("================================================================================================================")
print(list)
print(" ")

########################################################################################################################
#itung df
DfDict = dict.fromkeys(list[0].keys(), 0)
for doc in list:
    for word, val in doc.items():
        if val > 0:
            DfDict[word] += 1

print("ini DF Dictonary")
print(DfDict)
print("")

#FEATURE SELECTION
print("")
minthreshold = input("MASUKKAN MIN THRESHOLD  : ")
print("")
maxthreshold = input("MASUKKAN MAX THRESHOLD  : ")
print("")
min_threshold = int(minthreshold)
max_threshold = int(maxthreshold)
new_dict = {key: val for key, val in DfDict.items() if val >= min_threshold and val <= max_threshold}
print("new dict")
print(new_dict)
print("")

def word_count(str):
    w = dict.fromkeys(new_dict,0)
    words = str.split()

    for word in words:
        if word in w:
            w[word] += 1
    return w

list = []
for kalimat in preprocessing_doc:
    hasil = word_count(kalimat)
    list.append(hasil)
print("ini list")
print(list)


def dataFrame(hasil):
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    test = pd.DataFrame(hasil)
    return test
print("================================================================================================================")
print("Tampilan Gambaran Matriks Hasil Word Count")
print("================================================================================================================")
print(dataFrame(list))
print(" ")

#compute TF
import math
def computeTF(wordDict):
    tfDict = {}
    for word, count in wordDict.items():
        if count == 0:
            tfDict[word] = 0
        else:
            tfDict[word] = 1 + math.log10(float(count))
    return tfDict
#eksekusi compute TF
print("================================================================================================================")
print("Hasil Eksekusi Compute TF")
print("================================================================================================================")
list_tf = []
for dicWord in list:
    hasil_TF = computeTF(dicWord)
    print(hasil_TF)
    list_tf.append(hasil_TF)
print(" ")

#membuat compute IDF
def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(),0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] +=1
    #print("ini idfDict")
    #print(idfDict)
    #print("")
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N/float(val))


    return idfDict
print("================================================================================================================")
print("Hasil Itung IDF")
print("================================================================================================================")
idfs = computeIDF(list)
print(idfs)
print(" ")

#membuat compute TFIDF
def computeTFIDF(tfBow,idfs):
    tfidf = {}
    for words, val in  tfBow.items():
        tfidf[words] = val*idfs[words]

    return tfidf

list_tfidf = []
#menghitung hasil tf dikali idfs
for dicWord in list_tf:
    hasil_tfidf = computeTFIDF(dicWord,idfs)
    print(hasil_tfidf)
    list_tfidf.append(hasil_tfidf)
print("================================================================================================================")
print("List Hasil Itung TFIDF")
print("================================================================================================================")
print(list_tfidf)
print(" ")


import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

test2 = pd.DataFrame(list_tfidf)
print("================================================================================================================")
print("Ini Gambaran matriks hasil TFIDF")
print("================================================================================================================")
print("")
print(test2)
print("")

print("test")
print(test2.values)
print("")

########################################################################################################################

#try minmaxscalar

#from sklearn.preprocessing import MinMaxScaler

#x = test2.values
#min_max_scaler = MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#dataset = pd.DataFrame(x_scaled)

#print(dataset)

########################################################################################################################
#try normalize
import numpy as np
from sklearn.preprocessing import normalize
norm = normalize(test2.values)
#df = pd.DataFrame(np.concatenate(norm)
print("================================================================================================================")
print("Hasil Normalization")
print("================================================================================================================")
print(norm)
print("")

print("test som")
import test_som

#dataTest = [[0.1, 0.2, 0.3, 0.4],
#            [0.5, 0.6, 0.7, 0.8],
#            [0.9, 0.10, 0.11, 0.12]]

som = test_som.selfOrganizingMaps(norm,0.6, 0.5, 20)
print(som)