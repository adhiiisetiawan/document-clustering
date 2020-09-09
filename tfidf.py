import preprocessing

def tfIdfFunction(document):
    preprocessing_doc = []
    for d in document:
        preprocessing_doc.append(preprocessing.text_preprocessing(d))

    # membuat daftar bag of words yang isinya kata2 displit setiap dokumen
    list_bow = []
    for d in range(len(preprocessing_doc)):
        bow = preprocessing_doc[d].split(" ")
        list_bow.append(bow)

    # untuk membuat set kata (kata unik)
    preprocessing_doc = []
    for d in document:
        preprocessing_doc.append(preprocessing.text_preprocessing(d))
    gabungan = " ".join(preprocessing_doc)
    kata2 = gabungan.split()
    wordset = set(kata2)

    # membuat method pengitung kata2
    def word_count(str):
        w = dict.fromkeys(wordset, 0)
        words = str.split()

        for word in words:
            if word in w:
                w[word] += 1
            else:
                w[word] = 1

        return w

    # append dictonary ke list
    list = []
    for kalimat in preprocessing_doc:
        hasil = word_count(kalimat)
        list.append(hasil)

    # itung df
    DfDict = dict.fromkeys(list[0].keys(), 0)
    for doc in list:
        for word, val in doc.items():
            if val > 0:
                DfDict[word] += 1
    print(DfDict)

    # FEATURE SELECTION
    print("")
    minthreshold = input("MASUKKAN MIN THRESHOLD  : ")
    print("")
    maxthreshold = input("MASUKKAN MAX THRESHOLD  : ")
    print("")
    min_threshold = int(minthreshold)
    max_threshold = int(maxthreshold)
    new_dict = {key: val for key, val in DfDict.items() if val >= min_threshold and val <= max_threshold}
    print(new_dict)

    def word_count(str):
        w = dict.fromkeys(new_dict, 0)
        words = str.split()

        for word in words:
            if word in w:
                w[word] += 1
        return w

    list = []
    for kalimat in preprocessing_doc:
        hasil = word_count(kalimat)
        list.append(hasil)

    # def dataFrame(hasil):
    #     import pandas as pd
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.max_rows', None)
    #     test = pd.DataFrame(hasil)
    #     return test

    # compute TF
    import math
    def computeTF(wordDict):
        tfDict = {}
        for word, count in wordDict.items():
            if count == 0:
                tfDict[word] = 0
            else:
                tfDict[word] = 1 + math.log10(float(count))
        return tfDict

    # eksekusi compute TF
    list_tf = []
    for dicWord in list:
        hasil_TF = computeTF(dicWord)
        list_tf.append(hasil_TF)

    # membuat compute IDF
    def computeIDF(docList):
        import math
        idfDict = {}
        N = len(docList)
        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for doc in docList:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1
        # print("ini idfDict")
        # print(idfDict)
        # print("")
        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))
        #
        #
        return idfDict

    idfs = computeIDF(list)

    # membuat compute TFIDF
    def computeTFIDF(tfBow, idfs):
        tfidf = {}
        for words, val in tfBow.items():
            tfidf[words] = val * idfs[words]

        return tfidf

    list_tfidf = []
    # menghitung hasil tf dikali idfs
    for dicWord in list_tf:
        hasil_tfidf = computeTFIDF(dicWord, idfs)
        # print(hasil_tfidf)
        list_tfidf.append(hasil_tfidf)

    return list_tfidf