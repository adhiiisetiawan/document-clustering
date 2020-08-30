import preprocessing
import numpy as np

#docA = "Layang-layang terbang diangkasa"
#docB = "Burung- burung terbang diangkasa"
#docC = "Banyak layang-Layang berbentuk burung"
#docD = "Burung-burung di angkasa pulang di sore hari"
#docE = "Burung terbang untuk pulang ke sarang"

document = [
    "Layang-layang terbang diangkasa",
    "Burung- burung terbang diangkasa",
    "Banyak layang-Layang berbentuk burung",
    "Burung-burung di angkasa pulang di sore hari",
    "Burung terbang untuk pulang ke sarang"
]
print("Ini Dokumen2")
print(document)
print("")
preprocessing_doc = []

for d in document:
    preprocessing_doc.append(preprocessing.text_preprocessing(d))

print("Dokumen setelah preprocessing lansung")
print(preprocessing_doc)



stringDocument = " "
documentToString = stringDocument.join(preprocessing_doc)
bow = documentToString.split(" ")
print("")
print("hasil split adhi")
print(bow)

wordset = set(bow)
print("")
print("hasil remove split")
print(wordset)

wordDict = dict.fromkeys(wordset,0)

for word in bow:
    wordDict[word] +=1
print("")
print("word dict adhi")
print(wordDict)

def computeTF(wordDict,bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

tfBow = computeTF(wordDict,bow)
print("")
print("Hasil tf adhi")
print(tfBow)

# disini remove duplicate kuhapus
# removeDuplicate = list(
#     dict.fromkeys(hasilSplit))
# print("remove duplicate")
# print(removeDuplicate)
#
# string = " "  # membuat sebuah varible kosong bernama string dengan tipe string juga yang nantinya akan digunakan untuk mengkonversi list ke string
# hasilRemove = string.join(removeDuplicate)
# print(hasilRemove)


print("")
print("Ini Dokumen2 setelah preprocessing")
print(preprocessing_doc[0])
print(preprocessing_doc[1])
print(preprocessing_doc[2])
print(preprocessing_doc[3])
print(preprocessing_doc[4])
print("")

bowDoc = ""
print("hasil Bow Doc")
for i in preprocessing_doc:
    bowDoc = i.split(" ")
    # bowDoc = doc.split(" ")
    # print(bowDoc)
    hasilbow = set(bowDoc)
    # print(set(hasilbow))
    wordDictA = dict.fromkeys(hasilbow, 0)
    # print(bowDoc)
    # for word in bowDoc:
    wordDictA[bowDoc] += 1
    print(wordDictA)
print("")
# print("worddict adhi yg kedua")
# for word in hasilbow:


print("")
docA = preprocessing_doc[0]
docB = preprocessing_doc[1]
docC = preprocessing_doc[2]
docD = preprocessing_doc[3]
docE = preprocessing_doc[4]



bowA = docA.split(" ")
bowB = docB.split(" ")
bowC = docC.split(" ")
bowD = docD.split(" ")
bowE = docE.split(" ")
print("Ini dokumentasi2 setelah displit setiap dokumen")
print(bowA)
print(bowB)
print(bowC)
print(bowD)
print(bowE)



#print(bowA)
#print(bowB)
print("")

wordSet = set(bowA).union(set(bowB),set(bowC),set(bowD),set(bowE))

print("Ini WordSet")
print(wordSet)

wordDictA = dict.fromkeys(wordSet,0)
wordDictB = dict.fromkeys(wordSet,0)
wordDictC = dict.fromkeys(wordSet,0)
wordDictD = dict.fromkeys(wordSet,0)
wordDictE = dict.fromkeys(wordSet,0)

print("")
#print(wordDictA)
#print(wordDictB)

for word in bowA:
    wordDictA[word]+=1

for word in bowB:
    wordDictB[word]+=1

for word in bowC:
    wordDictC[word]+=1

for word in bowD:
    wordDictD[word]+=1

for word in bowE:
    wordDictE[word]+=1

print("Ini wordDict")
print(wordDictA)
print(wordDictB)
print(wordDictC)
print(wordDictD)
print(wordDictE)
print("")

import pandas as pd

test = pd.DataFrame([wordDictA,wordDictB,wordDictC,wordDictD,wordDictE])

print("Ini Gambaran matriks")
print(test)

tfBowA = computeTF(wordDictA,bowA)
tfBowB = computeTF(wordDictB,bowB)
tfBowC = computeTF(wordDictC,bowC)
tfBowD = computeTF(wordDictD,bowD)
tfBowE = computeTF(wordDictE,bowE)

print("")
print("Ini Hasil TF")
print(tfBowA)
print(tfBowB)
print(tfBowC)
print(tfBowD)
print(tfBowE)

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
        idfDict[word] = math.log(N/float(val))

    return idfDict

idfs = computeIDF([wordDictA, wordDictB, wordDictC, wordDictD, wordDictE])

print("")
print("ini idfs")
print(idfs)



def computeTFIDF(tfBow,idfs):
    tfidf = {}
    for words, val in  tfBow.items():
        tfidf[words] = val*idfs[words]

    return tfidf

tfidfbowA = computeTFIDF(tfBowA,idfs)
tfidfbowB = computeTFIDF(tfBowB,idfs)
tfidfbowC = computeTFIDF(tfBowC,idfs)
tfidfbowD = computeTFIDF(tfBowD,idfs)
tfidfbowE = computeTFIDF(tfBowE,idfs)

print("")
print("Ini Hasil TFIDF setiap Bag of words")
print(tfidfbowA)
print(tfidfbowB)
print(tfidfbowC)
print(tfidfbowD)
print(tfidfbowE)

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

test2 = pd.DataFrame([tfidfbowA,tfidfbowB,tfidfbowC,tfidfbowD,tfidfbowE])
print("")
print("Ini Gambaran matriks hasil TFIDF")
print(test2)