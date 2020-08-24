import preprocessing, tfidf, som

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    document = "Pembatasan Sosial Berskala Besar di ibu kota hampir berakhir—dan sepertinya benar-benar tak bakal " \
               "diperpanjang—seiring bergaungnya tatanan hidup baru yang disebut new normal. " \
               "Presiden Joko Widodo—dan WHO—telah menyatakan secara eksplisit bahwa kita, umat manusia, " \
               "harus siap hidup bersama virus corona—untuk selamanya, anggaplah begitu untuk saat ini karena obat maupun " \
               "vaksin corona belum selesai diracik.  PSBB selama ini menjadi salah satu strategi pemerintah RI untuk menghambat " \
               "laju penularan virus corona. PSBB mensyaratkan warga untuk untuk bekerja dan belajar dari rumah, tak keluar rumah " \
               "untuk urusan yang tak mendesak, dan menghindari kontak fisik serta menjaga jarak dengan orang lain. Namun, seperti segalanya, " \
               "semua masa ada ujungnya. Pula PSBB."

    dataTest = "Indonesia mengalami masa psbb yang sangat lama, yaitu sekitar 3 bulan. psbb dilakukan untuk mencegah" \
               "penyebaran virus corona."

    preprocessing_doc = preprocessing.text_preprocessing(document)
    document_weighting = tfidf.tfIdfCalculation([preprocessing_doc])

    preprocessing_dataTest = preprocessing.text_preprocessing(dataTest)
    document_weighting_dataTest = tfidf.tfIdfCalculation([preprocessing_dataTest])
    print(document_weighting_dataTest)
    som = som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 1, document_weighting_dataTest)
    print(som)