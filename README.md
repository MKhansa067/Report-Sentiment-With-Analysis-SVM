# Laporan: Analisis Sentimen SVM  
**1. Pendahuluan**  
Analisis sentimen adalah proses otomatisasi untuk menentukan opini atau perasaan pengguna terhadap suatu topik melalui teks. Salah satu metode yang banyak digunakan dalam analisis sentimen adalah Support Vector Machine (SVM). Pada pengembangan aplikasi machine learning yang kompleks dan berskala besar, efisiensi dan kontinuitas dalam proses pelatihan model menjadi hal yang krusial. Salah satu pendekatan yang umum diterapkan adalah dengan menggunakan sistem checkpoint, di mana model yang telah dilatih sebagian atau sepenuhnya dapat disimpan dan dimuat kembali tanpa perlu melakukan pelatihan ulang dari awal. Laporan ini membahas implementasi SVM untuk klasifikasi sentimen menggunakan dua skrip Python yaitu: sentiment_analysis_SVM.py dengan dataset yang digunakan adalah dataset dari file test.csv dan train.csv, dan SVM-checkpoint.py dengan dataset yang digunakan adalah Polarity Dataset v2.0 dari Cornell University dan dataset dari file Training.txt.  

**2. Rangkaian Proses Analisis pada Skrip sentiment_analysis_SVM.py**  
Skrip sentiment_analysis_SVM.py menyajikan pipeline sederhana untuk membuat model klasifikasi sentimen berbasis SVM. Berikut ini adalah tahapan-tahapan utama dalam implementasinya:  

**2.1. Load Dataset**  
Langkah pertama adalah memuat dua dataset, yaitu train.csv dan test.csv, yang masing-masing berisi data pelatihan dan data pengujian. Setiap baris dalam dataset terdiri dari teks ulasan dan label sentimen (positif/negatif).  

trainData = pd.read_csv("train.csv")  
testData = pd.read_csv("test.csv")  

**2.2. Ekstraksi Fitur dengan TF-IDF**  
Agar teks dapat diproses oleh algoritma pembelajaran mesin, data harus diubah ke dalam bentuk vektor numerik. Dalam kasus ini, digunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency) yang memberikan bobot pada kata-kata berdasarkan frekuensi dan eksklusivitasnya.  

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)  
train_vectors = vectorizer.fit_transform(trainData['Content'])  
test_vectors = vectorizer.transform(testData['Content'])  

Penjelasan Parameter:
- min_df=5: Mengabaikan kata yang muncul kurang dari lima kali.  
- max_df=0.8: Mengabaikan kata yang muncul di lebih dari 80% dokumen.  
- sublinear_tf=True: Mengubah frekuensi term menggunakan logaritma.  
- use_idf=True: Mengaktifkan skema pembobotan IDF.

**2.3. Pelatihan Model dengan SVM**  
Setelah fitur diperoleh, dilakukan pelatihan model SVM menggunakan kernel linear, yang cocok untuk data teks yang linier separable.  

classifier_linear = svm.SVC(kernel='linear')  
classifier_linear.fit(train_vectors, trainData['Label'])  
prediction_linear = classifier_linear.predict(test_vectors)  

Model dilatih pada train_vectors dan Label, kemudian digunakan untuk memprediksi label dari test_vectors.  

**2.4. Evaluasi Model**  
Evaluasi performa dilakukan menggunakan waktu pelatihan dan prediksi serta laporan klasifikasi (classification report) yang mencakup metrik seperti precision, recall, dan f1-score.  

print("Results for SVC(kernel=linear)")  
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))  
report = classification_report(testData['Label'], prediction_linear, output_dict=True)  
print('positive: ', report['pos'])  
print('negative: ', report['neg'])  

Hasil menunjukkan bahwa model SVM dengan kernel linear dapat membedakan sentimen positif dan negatif dengan cukup baik. Waktu pelatihan dan prediksi juga relatif efisien, menjadikan pendekatan ini cocok untuk eksperimen awal dalam klasifikasi teks.  

**3. Alur Kerja Skrip SVM-checkpoint.py**
Skrip SVM-checkpoint.py disusun dengan logika kerja utama sebagai berikut:  
**3.1. Load Dataset**  
Dataset pelatihan dan pengujian dimuat dari dua file CSV, masing-masing train.csv dan test.csv. Dataset ini berisi kolom Content (teks ulasan) dan Label (sentimen).  

trainData = pd.read_csv("train.csv")  
testData = pd.read_csv("test.csv")  

**3.2. Vektorisasi Teks dengan TF-IDF**  
Untuk dapat diproses oleh model SVM, data teks diubah menjadi representasi numerik menggunakan teknik TF-IDF.  

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)  
train_vectors = vectorizer.fit_transform(trainData['Content'])  
test_vectors = vectorizer.transform(testData['Content'])  

**3.3. Implementasi Checkpoint**  
Bagian ini merupakan inti dari skrip. Checkpoint dilakukan dengan memanfaatkan pustaka joblib untuk menyimpan dan memuat model yang telah dilatih.  

if os.path.exists("svm_model_checkpoint.joblib"):  
    classifier_linear = joblib.load("svm_model_checkpoint.joblib")  
else:  
    classifier_linear = svm.SVC(kernel='linear')  
    classifier_linear.fit(train_vectors, trainData['Label'])  
    joblib.dump(classifier_linear, "svm_model_checkpoint.joblib")  

Penjelasan:  
- Jika file checkpoint (svm_model_checkpoint.joblib) tersedia, maka model akan dimuat.  
- Jika belum tersedia, model akan dilatih dan disimpan dalam bentuk file .joblib.

**3.4. Prediksi dan Evaluasi Model**  
Model yang telah dimuat atau dilatih kemudian digunakan untuk memprediksi sentimen pada data uji, dan performa model dievaluasi dengan metrik klasifikasi.  

prediction_linear = classifier_linear.predict(test_vectors)  
report = classification_report(testData['Label'], prediction_linear, output_dict=True)  
print('positive: ', report['pos'])  
print('negative: ', report['neg'])  

Hasil evaluasi menunjukkan akurasi yang sebanding dengan pelatihan penuh, namun dengan efisiensi waktu yang meningkat ketika model dimuat dari checkpoint.  

**4. Kesimpulan**  
Melalui dua skrip yang diuji, yakni sentiment_analysis_SVM.py dan SVM-checkpoint.py, dapat disimpulkan bahwa pendekatan klasifikasi sentimen menggunakan algoritma SVM terbukti cukup efektif dalam membedakan opini positif dan negatif dari sebuah teks ulasan. Keduanya menggunakan teknik TF-IDF untuk mengubah teks menjadi format numerik yang bisa diproses oleh model.  

Secara keseluruhan, kedua skrip ini tidak hanya menunjukkan efektivitas SVM dalam tugas analisis sentimen, tetapi juga menggarisbawahi pentingnya efisiensi dalam proses pengembangan model machine learning. Untuk kebutuhan praktis dan berkelanjutan, penerapan checkpoint sangat disarankan karena mampu menghemat waktu dan sumber daya tanpa mengorbankan kualitas prediksi.  
