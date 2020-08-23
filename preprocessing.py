#mengimport regex
import re

# disini saya menggunakan pySastrawi dalam hal stemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def text_preprocessing(document):
    # pada bagian ini digunakan untuk casefolding
    caseFolding = document.lower()

    # ***ini digunakan untuk tokenisasi, disini saya menggunakan regex untuk
    # membersihkan tanda baca yang sekiranya mengganggu ***
    tokenization = re.findall(r"[\w']+", caseFolding)

    # proses stopwords removal dimulai dari sini
    file = open('stopword_tala.txt', 'r')  # disini saya membuka dokumen stopword tala
    stopWordsList = file.read()  # kemudian membacanya dan disimpan pada variable stopwordsList
    hasilStopwords = []  # saya membuat sebuah list kosong yang nantinya akan disimpan sebuah hasil dari stopwords

    for w in tokenization:  # melakukan perulangan variable w pada hasil tokenisasi
        if w not in stopWordsList:  # memberikan seleksi kondisi jika nilai w tidak ada dalam stopwordsList
            hasilStopwords.append(w)  # nilai dari w yang sudah diseleksi pada baris sebelumnya akan dimasukkan ke variable hasilStopwords
    # Stopword Removal
    # removeDuplicate = list(
    #     dict.fromkeys(hasilStopwords))  # menghapus duplikasi kata dalam list pada variable hasilStopwords

    string = " "  # membuat sebuah varible kosong bernama string dengan tipe string juga yang nantinya akan digunakan untuk mengkonversi list ke string
    stopwordListToString = string.join(hasilStopwords)  # menggabungkan string pada list

    # membuat stemmer dari library pySastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # proses stemming yang digunakan
    hasilStemming = stemmer.stem(stopwordListToString)

    return hasilStemming