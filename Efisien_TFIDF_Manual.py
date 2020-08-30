document = [
    "Layang-layang terbang diangkasa",
    "Burung- burung terbang diangkasa",
    "Banyak layang-Layang berbentuk burung",
    "Burung-burung di angkasa pulang di sore hari",
    "Burung terbang untuk pulang ke sarang",
]
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
#tampilan gambaran matriks hasil word count
def dataFrame(hasil):
    import pandas as pd
    test = pd.DataFrame(hasil)
    return test
print("================================================================================================================")
print("Tampilan Gambaran Matriks Hasil Word Count")
print("================================================================================================================")
print(dataFrame(list))
print(" ")
#######################################################################################################################
#compute TF
import math
def computeTF(wordDict):
    tfDict = {}
    for word, count in wordDict.items():
        #rumusnya masih count, math.log error entah kenapa
        # print(count)
        if count == 0:
            tfDict[word] = 0
        else:
            tfDict[word] = 1 + math.log10(float(count))
    return tfDict
#eksekusi compute TF
print("================================================================================================================")
print("Hasil Eksekusi Compute TF (Masih ada kurang rumusnya TF Weighting)")
print("================================================================================================================")
for dicWord in list:
    print(computeTF(dicWord))
print(" ")
########################################################################################################################
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


    for word, val in idfDict.items():
        idfDict[word] = math.log10(N/float(val))


    return idfDict
print("================================================================================================================")
print("Hasil Itung IDF")
print("================================================================================================================")
idfs = computeIDF(list)
print(idfs)
print(" ")

########################################################################################################################
#membuat compute TFIDF
def computeTFIDF(tfBow,idfs):
    tfidf = {}
    for words, val in  tfBow.items():
        tfidf[words] = val*idfs[words]

    return tfidf
list_tfidf = []
#menghitung hasil tf dikali idfs
for dicWord in list:
    hasil_tfidf = computeTFIDF(dicWord,idfs)
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