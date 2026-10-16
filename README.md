# Panduan Penggunaan Champion Template Notebook (Final Data Mining)

Direktori ini menyediakan dua template notebook yang siap di-upload ke Kaggle untuk memenangkan kompetisi data mining di bawah kendala **1 Workstation, 3 Anggota Tim, dan Durasi Singkat (< 4 Jam)**.

---

## 📁 Pilihan Template Notebook di Folder Ini

### 1. Template 1: Fast Robust GBDT Blend (Kecepatan & Kestabilan)
- **Notebook:** [`tabular_mining_champion_template.ipynb`](file:///d:/vscode/tecpo-f/tabular_mining_champion_template.ipynb)
- **Script:** [`champion_pipeline.py`](file:///d:/vscode/tecpo-f/champion_pipeline.py)
- **Metode:** 5-Fold Stratified CV, rekayasa fitur frekuensi & grup, Trio GBDT (LightGBM + CatBoost + XGBoost), dan SLSQP dynamic weight optimization.
- **Karakter:** Ringan, waktu latih ~35 detik, sangat minim risiko, fokus pada kecepatan eksekusi.

### 2. Template 2: Grandmaster Multi-Layer Stacking (Fokus Juara 1)
- **Notebook:** [`grandmaster_tabular_champion.ipynb`](file:///d:/vscode/tecpo-f/grandmaster_tabular_champion.ipynb)
- **Script:** [`grandmaster_stacking_pipeline.py`](file:///d:/vscode/tecpo-f/grandmaster_stacking_pipeline.py)
- **Metode Grandmaster:**
  1. **Adversarial Validation (AV):** Menguji apakah distribusi data train dan test identik untuk mendeteksi *covariate shift*.
  2. **Feature Engineering Factory:** Statistik baris (*row summaries*), K-Means cluster distance (menembus batas non-linear pohon), PCA variance projections, interaksi kategori.
  3. **Level-1 Model Zoo (4 Model):** LightGBM (leaf-wise), CatBoost (oblivious trees), XGBoost (depth-wise), dan LogisticRegression/Ridge (linear baseline).
  4. **Level-2 Stacking Meta-Learner:** Model meta regresi linear ter-regularisasi (L2) yang dilatih pada matriks prediksi Out-of-Fold (OOF).
  5. **Rank-Averaged Ensemble Fallback:** Mengatasi ketidakcocokan kalibrasi probabilitas antar model.
- **Karakter:** Metodologi paling canggih, waktu latih ~40 detik, nilai sempurna untuk Ketepatan Metode (25%) dan skor Private Leaderboard (35%).

---

## ⏱️ Pembagian Tugas Tim (Total 210 Menit)

| Waktu | Pilot (Di Workstation) | Storyteller (Di Kertas / Meja) | Methodologist (Di Kertas / Meja) |
|---|---|---|---|
| **00:00 - 00:30** | Buka Kaggle, upload notebook, sesuaikan `CONFIG`, jalankan baseline. | Baca panduan slide panitia. Tulis kerangka narasi masalah bisnis. | Analisis data dictionary. Amati hasil Adversarial Validation (apakah train dan test bergeser). |
| **00:30 - 01:30** | Tambahkan fitur domain khusus ke fungsi Feature Factory. | Mulai menyusun teks latar belakang, EDA, dan arsitektur di kertas. | Evaluasi matriks korelasi fitur dan periksa bobot Level-2 Meta-Learner. |
| **01:30 - 02:00** | **Model Freeze:** Eksekusi Level-2 Stacking & ekspor aset gambar ke zip. | Review draft slide dan siapkan layout untuk menempel gambar. | Analisis Confusion Matrix / Error Analysis dan siapkan naskah tanya-jawab juri. |
| **02:00 - 03:15** | Mengalah dari komputer. Bantu review teks slide. | **Ambil alih workstation:** Masukkan 5 gambar dari zip ke slide panitia. | Membantu pengetikan poin rekomendasi bisnis pada slide. |
| **03:15 - 03:30** | Submit 2 file (Stacking & Rank Average) ke Kaggle. | Gladi bersih presentasi (latihan bicara 7-10 menit). | Finalisasi argumen metodologi dan simulasi Q&A juri. |

---

## ⚙️ Cara Mengatur `CONFIG` di Notebook

Pada **Sel 2**, sesuaikan parameter target:
```python
CONFIG = {
    "TRAIN_PATH": "/kaggle/input/nama-kompetisi/train.csv",
    "TEST_PATH": "/kaggle/input/nama-kompetisi/test.csv",
    "SAMPLE_SUB_PATH": "/kaggle/input/nama-kompetisi/sample_submission.csv",
    "EXPORT_DIR": "./presentation_assets_grandmaster/",
    "ID_COL": "id",
    "TARGET_COL": "target",
    "TASK": "binary",         # 'binary', 'multiclass', 'regression'
    "METRIC": "roc_auc",      # 'roc_auc', 'f1_macro', 'f1', 'accuracy', 'rmse', 'mae'
    "N_SPLITS": 5,
    "SEED": 42
}
```

---

## 📸 5 Gambar Otomatis untuk Slide Presentasi Grandmaster

Semua plot tersimpan otomatis di `./presentation_assets_grandmaster/` (300 DPI) dan dibundel ke `presentation_assets_grandmaster.zip`:
1. `01_adversarial_validation_drift.png`: ROC Curve uji pergeseran distribusi data Train vs Test (Bukti validasi juri).
2. `02_feature_space_clusters.png`: Visualisasi sebaran ruang klaster K-Means & proyeksi PCA.
3. `03_stacking_weights_and_gains.png`: Diagram batang bobot meta-learner dan perbandingan performa Level-1 vs Level-2.
4. `04_confusion_matrix_stacked.png`: Confusion Matrix atau Residual Plot dari ensemble stacking.
5. `05_shap_beeswarm.png`: Plot dampak arah fitur global (SHAP Beeswarm) untuk rekomendasi bisnis.

---

## 🛡️ Trik Menjawab Pertanyaan Juri (Kunci Nilai 25% Metodologi)
- **Tanya:** *"Mengapa Anda yakin model ini tidak overfit di Private Leaderboard?"*
  - **Jawab:** *"Pertama, kami melakukan Adversarial Validation untuk membuktikan bahwa data train dan test berasal dari distribusi yang sama (AUC mendekati 0.5). Kedua, skema validasi kami adalah 5-Fold Stratified Cross-Validation bebas kebocoran. Ketiga, model meta-learner Level-2 kami adalah model linear ter-regularisasi (L2 penalty) yang dilatih strictly pada data Out-of-Fold, sehingga secara matematis tidak dapat menghafal target."*
- **Tanya:** *"Mengapa menambahkan K-Means Cluster Distance dan PCA ke dalam model pohon?"*
  - **Jawab:** *"Model pohon keputusan (GBDT) memotong ruang fitur secara ortogonal (sejajar sumbu). Fitur jarak centroid K-Means dan komponen PCA memberikan batas keputusan radial dan kombinasi linear kontinu, memungkinkan pohon menangkap geometri data non-linear secara instan tanpa memerlukan kedalaman pohon yang berlebihan."*
