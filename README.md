# Panduan Penggunaan Champion Template Notebook (Final Data Mining)

Template ini dirancang khusus untuk kompetisi data mining final dengan kendala **1 Workstation, 3 Anggota Tim, dan Durasi Singkat (< 4 Jam)**.

---

## 📁 File Penting di Folder Ini
1. [`tabular_mining_champion_template.ipynb`](file:///d:/vscode/tecpo-f/tabular_mining_champion_template.ipynb): File Jupyter Notebook utama untuk di-upload langsung ke Kaggle (**File -> Upload Notebook**).
2. [`champion_pipeline.py`](file:///d:/vscode/tecpo-f/champion_pipeline.py): Versi script Python murni dengan penanda sel (`# %%`) yang siap dijalankan secara lokal atau di terminal.
3. `presentation_assets/`: Direktori output otomatis berisi gambar visualisasi beresolusi tinggi (300 DPI) untuk ditempel ke slide presentasi panitia.

---

## ⏱️ Pembagian Tugas Tim (210 Menit)

| Waktu | Pilot (Di Workstation) | Storyteller (Di Kertas / Meja) | Methodologist (Di Kertas / Meja) |
|---|---|---|---|
| **00:00 - 00:30** | Buka Kaggle, upload notebook, atur `CONFIG`, jalankan baseline. | Baca panduan & format slide panitia. Tulis kerangka narasi masalah bisnis. | Analisis kamus data (*data dictionary*). Cek kebocoran target & anomali data. |
| **00:30 - 01:30** | Terapkan fitur baru dari masukan Methodologist. Pantau skor OOF. | Mulai menyusun teks analisis latar belakang dan metodologi di kertas. | Periksa matriks korelasi dan buat hipotesis rekayasa fitur (*feature engineering*). |
| **01:30 - 02:00** | **Model Freeze:** Jalankan trio GBDT & blending. Ekspor aset gambar ke zip. | Review kelengkapan slide dan siapkan struktur untuk menempel gambar. | Lakukan audit kesalahan (*error analysis*) dan susun argumen metodologi. |
| **02:00 - 03:15** | Mengalah dari komputer. Bantu menyusun naskah kesimpulan dan limitasi. | **Ambil alih workstation:** Masukkan 5 gambar dari zip ke template slide panitia. | Membantu pengetikan poin rekomendasi bisnis pada slide. |
| **03:15 - 03:30** | Verifikasi submission file di Kaggle (format ID dan baris). | Gladi bersih presentasi (latihan berbicara 7-10 menit). | Antisipasi pertanyaan juri terkait validasi dan kebocoran data. |

---

## ⚙️ Cara Mengatur `CONFIG` di Notebook (30 Detik Pertama)

Pada **Sel 2**, sesuaikan parameter berikut:
```python
CONFIG = {
    # 1. Path Data (Sesuaikan path input di Kaggle)
    "TRAIN_PATH": "/kaggle/input/nama-kompetisi/train.csv",
    "TEST_PATH": "/kaggle/input/nama-kompetisi/test.csv",
    "SAMPLE_SUB_PATH": "/kaggle/input/nama-kompetisi/sample_submission.csv",
    "EXPORT_DIR": "./presentation_assets/",
    
    # 2. Nama Kolom
    "ID_COL": "id",           # Sesuaikan dengan nama kolom ID di dataset
    "TARGET_COL": "target",   # Sesuaikan dengan nama kolom target
    
    # 3. Tipe Tugas & Metrik
    "TASK": "binary",         # Pilihan: 'binary', 'multiclass', 'regression'
    "METRIC": "roc_auc",      # Pilihan: 'roc_auc', 'f1_macro', 'f1', 'accuracy', 'rmse', 'mae'
    
    # 4. Validasi
    "N_SPLITS": 5,            # Standar baku 5-fold cross-validation
    "SEED": 42
}
```

---

## 📸 5 Gambar Otomatis untuk Slide Presentasi (Bobot 30%)

Setelah notebook selesai dijalankan, buka folder `./presentation_assets/` atau unduh `presentation_assets.zip`:
1. `01_eda_target_distribution.png`: Masukkan ke slide **Exploratory Data Analysis (EDA)** untuk menunjukkan keseimbangan target.
2. `02_missing_values_matrix.png`: Masukkan ke slide **Data Quality Audit & Preprocessing**.
3. `03_cv_score_comparison.png`: Masukkan ke slide **Model Experimentation & Comparison** (menunjukkan keunggulan Ensemble atas single model).
4. `04_model_evaluation_matrix.png`: Masukkan ke slide **Error Analysis** (Confusion Matrix atau Residuals).
5. `05_shap_feature_importance.png`: Masukkan ke slide **Model Interpretability & Business Insights** (menjelaskan fitur apa yang paling memengaruhi keputusan).

---

## 🛡️ Trik Menjawab Pertanyaan Juri (Mengamankan 25% Nilai Metodologi)
- **Tanya:** *"Bagaimana Anda menjamin model tidak overfit terhadap leaderboard publik?"*
  - **Jawab:** *"Kami menggunakan 5-Fold Stratified Cross-Validation tanpa kebocoran data. Seluruh transformasi dan encoding dilakukan strictly di dalam masing-masing fold latih. Keputusan model sepenuhnya didasarkan pada skor Out-of-Fold (OOF), bukan skor public leaderboard."*
- **Tanya:** *"Mengapa memilih trio LightGBM, CatBoost, dan XGBoost?"*
  - **Jawab:** *"Ketiga model menggunakan arsitektur pemisahan pohon dan penanganan fitur kategorikal yang berbeda (LightGBM menggunakan leaf-wise, CatBoost menggunakan symmetric trees, XGBoost menggunakan depth-wise). Kombinasi rata-rata probabilitas ketiganya terbukti secara empiris meredam variansi prediksi dan mencegah private leaderboard shakeup."*
