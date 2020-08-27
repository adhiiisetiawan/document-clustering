import preprocessing, tfidf, som
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    document = [["Pengembangan Aplikasi Mobile Pendeteksi Penyakit Pada Tanaman Cabai Dengan Menggunakan Ximilar Custom Image Recognition (Studi Kasus: Balai Pengkajian Teknologi Pertanian, Kecamatan Karangploso, Kota Malang)",
"Pengembangan Aplikasi Data Warehouse Prestasi Mahasiswa (Studi Pada: Fakultas Ilmu Komputer Universitas Brawijaya)",
"Klasifikasi Emosi pada Komentar YouTube Menggunakan Metode Modified K-Nearest Neighbor (MKNN) dengan BM25 dan Seleksi Fitur Chi-Square",
"Perancangan Sistem Informasi Pembayaran Online pada Semester Antara Fakultas Ilmu Komputer Universitas Brawijaya",
"Pengembangan RestoCrowd: Aplikasi Android Penghitung Jumlah Pengunjung Restoran Berbasis Crowdsourcing dengan Ekstrapolasi",
"Mekanisme Load Balancing Server Menggunakan Metode Naive Bayes dengan Agen Psutils pada Software Defined Network",
"Perbandingan Usability Learning Management System Edmodo dan Google Classroom Menggunakan Metode Cognitive Walkthrough dan User Experience Questionnaire (UEQ) (Studi Kasus: SMKN 3 Malang)",
"Evaluasi dan Perbaikan Desain Antarmuka Pengguna Situs Web Indah Bordir Menggunakan Pendekatan Human Centered Design (HCD)",
"Sistem Prediksi Pertumbuhan Jumlah Penduduk Kota Malang menggunakan Metode K-Nearest Neighbor Regression",
"Sistem Pengontrol Presentasi Menggunakan Pengenalan Gestur Tangan Berbasis Fitur pada Contour dengan Metode Klasifikasi Support Vector Machine",
"Analisis Sentimen Berbasis Aspek pada Ulasan Pelanggan Restoran Bakso President Malang dengan Metode Naïve Bayes Classifier",
"Perancangan Dashboard Sistem Informasi Pemeringkatan UBAQA (UB Annual Quality Award) dengan Metode Human Centered Design",
"Perbaikan Desain Antarmuka Sistem Pelaporan Kehilangan Menggunakan Metode Human Centered Design (HCD) (Studi Kasus: SPKT Polres Tulungagung)",
"Evaluasi Proses Tata kelola Keamanan Informasi Menggunakan COBIT 5 Dengan Proses APO13, DSS04 dan DSS05 (Studi Pada DISKOMINFO Kabupaten Sidoarjo)",
"Analisis Sentimen Terhadap Ulasan Pengguna MRT Jakarta Menggunakan Information Gain dan Modified K-Nearest Neighbor",
"Pengembangan Aplikasi E-Commerce Menggunakan Payment Gateway Midtrans",
"Analisis Sentimen Berbasis Aspek Ulasan Pelanggan Terhadap Kertanegara Premium Guest House Menggunakan Support Vector Machine",
"Navigasi Robot Beroda Berdasarkan Pengenalan Teks untuk Melakukan Pergerakan Menggunakan Metode Optical Character Recognition (OCR)",
"Deteksi Pergerakan Bola Mata untuk Pemilihan Empat Menu Menggunakan Metode Facial Landmark dengan Ekstraksi Fitur LBP dan Klasifikasi K-NN",
"Sistem Presensi Mahasiswa Berdasarkan Pengenalan Wajah Menggunakan Metode LBP dan K-Nearest Neighbor Berbasis Mini PC",
"Pengaruh Dukungan Sosial Teman Sebaya dan Kedisiplinan Belajar Terhadap Hasil Belajar Siswa Kelas XI TKJ Mata Pelajaran Teknologi Jaringan Berbasis Luas (WAN) di SMK Negeri 6 Malang",
"Pengembangan Sistem Informasi Kredit Prestasi Berbasis Web (Studi Kasus: Fakultas Ekonomi dan Bisnis Universitas Brawijaya)",
"Pengaruh Kualitas Implementasi Metode Pembelajaran Ceramah Berbantuan Powerpoint dan Quizizz terhadap Hasil Belajar Kognitif dan Psikomotorik Mata Pelajaran Desain Grafis Percetakan di SMK Negeri 12 Malang",
"Evaluasi Dan Rekomendasi Perbaikan Website Virtual Learning Management Universitas Brawijaya pada Perangkat Bergerak Menggunakan Metode Heuristic Evaluation dan System Usability Scale (SUS)",
"Pengaruh Kualitas Implementasi Metode Pembelajaran Problem Based Instruction Terhadap Hasil Belajar Kognitif Dan Psikomotor Pada Mata Pelajaran Pemrograman Dasar Di SMK Negeri 3 Malang",
"Evaluasi Usability pada Aplikasi Perangkat Bergerak SIP Dispendukcapil Jember dengan Metode Heuristic Evaluation dan Usability Testing",
"Perancangan User Experience Aplikasi Pendukung Evaluasi dan Analisis Proses Pembelajaran untuk Guru Berbasis Android dengan Metode User-Centered Design dan Design Solution",
"Pengaruh Kualitas Implementasi Model Pembelajaran Tipe Student Teams Achievements Divisions (STAD) dan Model Pembelajaran Tipe Numbered Head Together (NHT) terhadap Hasil Belajar Siswa Kelas X Program Keahlian Teknik Komputer dan Informatika Mata Pelajaran",
"Pengembangan Sistem Manajemen Penjadwalan Les Privat Berbasis Web (Studi Kasus: Naoyuki Academic Center)",
"Pemodelan Arsitektur Bisnis Guna Mendukung Bisnis Berkelanjutan Menggunakan Pendekatan Enterprise Architecture (Studi Kasus: Kedai Kopi “Kopi Soe Malang”)",
"Prediksi Harga Emas Dengan Menggunakan Metode Average-Based Fuzzy Time Series",
"Evaluasi Usability dan Rekomendasi Perbaikan pada Aplikasi E-Kinerja Kabupaten Kediri menggunakan Metode Heuristic Evaluation",
"Pengembangan Sistem Manajemen Notulensi dan Dokumentasi Rapat Berbasis Web (Studi Kasus: Jurusan Teknik Informatika Fakultas Ilmu Komputer Universitas Brawijaya)",
"Pengembangan Sistem Monitoring Tingkat Stres berbasis Website",
"Temu Kembali Informasi Lintas Bahasa Dokumen Berita Bahasa Indonesia-Inggris menggunakan Metode BM25F",
"Prediksi Kecenderungan Pelanggan Telat Bayar pada Layanan Pembiayaan Adira Finance Saluran E-Commerce",
"Pengembangan Modul Digital Interaktif Berbasis Website menggunakan Kerangka Kerja Borg, Gall, And Gall pada Mata Pelajaran Administrasi Sistem Jaringan di SMK Negeri 12 Malang",
"Klasifikasi Jurusan Siswa menggunakan K-Nearest Neighbor dan Optimasi dengan Algoritme Genetika (Studi Kasus: SMAN 1 Wringinanom Gresik)",
"Analisis Pengalaman Pengguna Aplikasi Pemesanan Tiket Bioskop menggunakan User Experience Questionnaire (UEQ) dan Heuristic Evaluation (HE)",
"Evaluasi dan Perancangan User Experience menggunakan Metode Human Centered Design dan Heuristic Evaluation pada Aplikasi Dunia Games"]]

    preprocessing_doc = preprocessing.text_preprocessing(document)
    document_weighting = tfidf.tfIdfCalculation([preprocessing_doc])
    # preprocessing_doc = []
    #
    # for d in document:
    #     preprocessing_doc.append(preprocessing.text_preprocessing(d))

    # document_weighting = tfidf.tfIdfCalculation(preprocessing_doc)
    print(document_weighting)


    # som = som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 10)
    # print(som)

    # Visualisasi
    # X, target = make_blobs(n_samples=30, n_features=2, centers=3)
    # centroids = som.selfOrganizingMaps(document_weighting, 0.6, 0.5, 100)
    # colors = 'rgbcmyk'
    #
    # for x, label in zip(X, target):
    #     plt.plot(x[0], x[1], colors[label] + '.')
    #
    # plt.plot(centroids[:, 0], centroids[:, 1], 'kx')
    # plt.show()