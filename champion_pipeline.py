# %% [markdown]
# # [CHAMPION] Tabular Data Mining Competition: Champion Template Notebook
# **Tujuan:** Memaksimalkan Skor Private Leaderboard (35%) sekaligus mengamankan 55% nilai (30% Presentasi + 25% Ketepatan Metode).
# 
# ### [TIME] Jadwal Timeboxing Tim (Total < 4 Jam / 210 Menit):
# - **00:00 - 00:30 (Inisiasi & Baseline):** Pilot memuat data, setting CONFIG, jalankan baseline. Storyteller & Methodologist membedah masalah di kertas.
# - **00:30 - 01:30 (Feature Engineering Terarah):** Methodologist menyuplai ide fitur domain. Pilot mengaktifkan modul FE di bawah. Cek kenaikan skor OOF.
# - **01:30 - 02:00 (Model Freeze, Ensembling & Ekspor Aset):** Bekukan kode. Jalankan trio GBDT (LGBM, CatBoost, XGBoost) & generate semua gambar untuk slide.
# - **02:00 - 03:15 (Penyusunan Slide di Workstation):** Storyteller mengambil alih komputer, memasukkan aset gambar ke template slide panitia.
# - **03:15 - 03:30 (Gladi Bersih & Q&A Prep):** Latihan presentasi dan verifikasi file submission.

# %%
# ==========================================
# 1. IMPORTS & ENVIRONMENT SETUP
# ==========================================
# CRITICAL: Import model C-libraries first to prevent OpenMP TLS crash on Windows/Workstations
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import os
import gc
import re
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings('ignore')

# Set visual styling for presentation-grade plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

print("[OK] Semua library berhasil diimpor dengan sukses!")

# %%
# ==========================================
# 2. CENTRAL CONFIGURATION (SESUAIKAN DI SINI)
# ==========================================
CONFIG = {
    # 1. File Paths (Sesuaikan path file di Kaggle)
    "TRAIN_PATH": "train.csv",
    "TEST_PATH": "test.csv",
    "SAMPLE_SUB_PATH": "sample_submission.csv",
    "EXPORT_DIR": "./presentation_assets/",
    
    # 2. Metadata Kolom
    "ID_COL": "id",
    "TARGET_COL": "target",
    
    # 3. Karakteristik Tugas
    # Pilihan task: 'binary', 'multiclass', 'regression'
    "TASK": "binary",
    
    # Pilihan metric: 'roc_auc', 'f1_macro', 'f1', 'accuracy', 'log_loss', 'rmse', 'mae'
    "METRIC": "roc_auc",
    
    # 4. Skema Validasi
    "N_SPLITS": 5,
    "SEED": 42,
    
    # 5. Model Switch
    "USE_LGBM": True,
    "USE_CATBOOST": True,
    "USE_XGB": True,
    "EARLY_STOPPING_ROUNDS": 100,
    "N_ESTIMATORS": 1500,
}

# Buat direktori ekspor visualisasi slide
os.makedirs(CONFIG["EXPORT_DIR"], exist_ok=True)
print(f"[DIR] Folder aset presentasi siap: {CONFIG['EXPORT_DIR']}")

# %%
# ==========================================
# 3. UTILITY & EVALUATION ENGINE
# ==========================================
def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CONFIG["SEED"])

def calculate_metric(y_true, y_pred, metric_name, task="binary"):
    """Menghitung metrik performa sesuai target lomba."""
    if metric_name == "roc_auc":
        if task == "binary":
            return roc_auc_score(y_true, y_pred)
        elif task == "multiclass":
            return roc_auc_score(y_true, y_pred, multi_class='ovr')
    elif metric_name == "f1_macro":
        return f1_score(y_true, y_pred, average='macro')
    elif metric_name == "f1":
        return f1_score(y_true, y_pred)
    elif metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric_name == "log_loss":
        return log_loss(y_true, y_pred)
    elif metric_name == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == "mae":
        return mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError(f"Metrik {metric_name} belum terdaftar!")

def optimize_threshold_for_f1(y_true, y_prob):
    """Mencari threshold probabilitas terbaik pada OOF untuk memaksimalkan F1."""
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        score = f1_score(y_true, (y_prob >= thresh).astype(int), average='macro' if CONFIG["TASK"]=="multiclass" else 'binary')
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

print("[OK] Utility & Evaluation Engine siap digunakan.")

# %%
# ==========================================
# 4. DATA LOADING & DATA INTEGRITY AUDIT
# ==========================================
if not os.path.exists(CONFIG["TRAIN_PATH"]):
    print("[WARN] File asli belum ditemukan. Membuat synthetic dummy data untuk demonstrasi...")
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=2000, n_features=15, n_informative=10, random_state=42)
    df_train = pd.DataFrame(X_syn, columns=[f"feat_{i}" for i in range(15)])
    df_train[CONFIG["ID_COL"]] = range(1, len(df_train) + 1)
    df_train[CONFIG["TARGET_COL"]] = y_syn
    df_train["category_A"] = np.random.choice(["Type1", "Type2", "Type3"], size=len(df_train))
    df_train.iloc[0:20, 2] = np.nan
    
    df_test = df_train.copy().drop(columns=[CONFIG["TARGET_COL"]]).iloc[:500].reset_index(drop=True)
    df_test[CONFIG["ID_COL"]] = range(5000, 5000 + len(df_test))
else:
    df_train = pd.read_csv(CONFIG["TRAIN_PATH"])
    df_test = pd.read_csv(CONFIG["TEST_PATH"])

print(f"[DATA] Dimensi Train: {df_train.shape}")
print(f"[DATA] Dimensi Test : {df_test.shape}")

audit_df = pd.DataFrame({
    'Dtype': df_train.dtypes,
    'Null_Count': df_train.isnull().sum(),
    'Null_Pct': (df_train.isnull().sum() / len(df_train)) * 100
})
null_columns = audit_df[audit_df['Null_Count'] > 0].sort_values(by='Null_Count', ascending=False)
if len(null_columns) > 0:
    print("\n[AUDIT] Kolom dengan missing values tertinggi:")
    print(null_columns.head())
else:
    print("\n[OK] Tidak ada missing value pada data train.")

# %%
# ==========================================
# 5. AUTOMATED EDA & PRESENTATION PLOT EXPORTER
# ==========================================
# 1. Target Distribution Plot (Slide Bahan 1)
plt.figure(figsize=(7, 4))
if CONFIG["TASK"] in ["binary", "multiclass"]:
    ax = sns.countplot(data=df_train, x=CONFIG["TARGET_COL"], palette="mako")
    total = len(df_train)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}\n({height/total:.1%})',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', weight='bold')
    plt.title("Target Distribution & Balance (EDA)", fontsize=14, weight='bold')
else:
    sns.histplot(df_train[CONFIG["TARGET_COL"]], kde=True, color='teal')
    plt.title("Target Variable Distribution (EDA)", fontsize=14, weight='bold')

plt.tight_layout()
target_plot_path = os.path.join(CONFIG["EXPORT_DIR"], "01_eda_target_distribution.png")
plt.savefig(target_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan untuk slide: {target_plot_path}")

# 2. Missing Values Matrix Plot (Slide Bahan 2)
null_counts = df_train.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

if len(null_counts) > 0:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=null_counts.values, y=null_counts.index, palette="flare")
    plt.title("Missing Values Audit (EDA)", fontsize=14, weight='bold')
    plt.xlabel("Jumlah Data Kosong")
    plt.tight_layout()
    missing_plot_path = os.path.join(CONFIG["EXPORT_DIR"], "02_missing_values_matrix.png")
    plt.savefig(missing_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Tersimpan untuk slide: {missing_plot_path}")
else:
    print("[INFO] Tidak ditemukan missing value pada dataset.")

# %%
# ==========================================
# 6. MODULAR FEATURE ENGINEERING TOOLKIT
# ==========================================
def apply_feature_engineering(train_data, test_data, config):
    """
    Modul rekayasa fitur siap pakai.
    Target Encoding dilakukan terpisah di dalam loop CV untuk mencegah kebocoran data.
    """
    df_tr = train_data.copy()
    df_te = test_data.copy()
    
    target_col = config["TARGET_COL"]
    id_col = config["ID_COL"]
    
    features = [c for c in df_tr.columns if c not in [id_col, target_col]]
    cat_cols = [c for c in features if df_tr[c].dtype == 'object' or df_tr[c].dtype.name == 'category']
    num_cols = [c for c in features if c not in cat_cols]
    
    print(f"[FE] Kolom Kategori Terdeteksi ({len(cat_cols)}): {cat_cols}")
    print(f"[FE] Kolom Numerik Terdeteksi ({len(num_cols)}): {num_cols[:10]} ...")
    
    # 1. Frequency / Count Encoding untuk Kolom Kategori
    for c in cat_cols:
        freq = df_tr[c].value_counts().to_dict()
        df_tr[f"{c}_freq"] = df_tr[c].map(freq).fillna(0)
        df_te[f"{c}_freq"] = df_te[c].map(freq).fillna(0)
    
    # 2. Group Aggregation (Statistik numerik berdasarkan kategori utama)
    if len(cat_cols) > 0 and len(num_cols) > 0:
        main_cat = cat_cols[0]
        for num_c in num_cols[:3]:
            agg_stats = df_tr.groupby(main_cat)[num_c].agg(['mean', 'std']).reset_index()
            agg_stats.columns = [main_cat, f"{main_cat}_{num_c}_grp_mean", f"{main_cat}_{num_c}_grp_std"]
            
            df_tr = df_tr.merge(agg_stats, on=main_cat, how='left')
            df_te = df_te.merge(agg_stats, on=main_cat, how='left')
            
    # 3. Label Encoding untuk sisa kolom kategori
    for c in cat_cols:
        le = LabelEncoder()
        all_vals = list(df_tr[c].astype(str).values) + list(df_te[c].astype(str).values)
        le.fit(all_vals)
        df_tr[c] = le.transform(df_tr[c].astype(str))
        df_te[c] = le.transform(df_te[c].astype(str))
        
    return df_tr, df_te

df_train_fe, df_test_fe = apply_feature_engineering(df_train, df_test, CONFIG)
print(f"[OK] Feature Engineering selesai. Dimensi baru train: {df_train_fe.shape}")

# %%
# ==========================================
# 7. LEAK-FREE 5-FOLD CROSS-VALIDATION ENGINE
# ==========================================
features = [c for c in df_train_fe.columns if c not in [CONFIG["ID_COL"], CONFIG["TARGET_COL"]]]
X = df_train_fe[features]
y = df_train_fe[CONFIG["TARGET_COL"]]
X_test = df_test_fe[features]

if CONFIG["TASK"] in ["binary", "multiclass"]:
    kf = StratifiedKFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["SEED"])
    split_gen = list(kf.split(X, y))
else:
    kf = KFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["SEED"])
    split_gen = list(kf.split(X))

oof_lgb = np.zeros(len(df_train_fe)) if CONFIG["TASK"] == "binary" else np.zeros((len(df_train_fe), len(np.unique(y))))
test_lgb = np.zeros(len(df_test_fe)) if CONFIG["TASK"] == "binary" else np.zeros((len(df_test_fe), len(np.unique(y))))

oof_cat = np.zeros_like(oof_lgb)
test_cat = np.zeros_like(test_lgb)

oof_xgb = np.zeros_like(oof_lgb)
test_xgb = np.zeros_like(test_lgb)

cv_scores_lgb = []
cv_scores_cat = []
cv_scores_xgb = []

saved_models = {"lgb": [], "cat": [], "xgb": []}

print(f"[START] Memulai {CONFIG['N_SPLITS']}-Fold Cross-Validation...")
print(f"[METRIC] Target Evaluasi Utama: Out-of-Fold (OOF) {CONFIG['METRIC'].upper()}")

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(split_gen):
    print(f"\n--- [FOLD {fold + 1} / {CONFIG['N_SPLITS']}] ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # ----------------- 1. LIGHTGBM -----------------
    if CONFIG["USE_LGBM"]:
        lgb_params = {
            'objective': 'binary' if CONFIG["TASK"]=='binary' else ('multiclass' if CONFIG["TASK"]=='multiclass' else 'regression'),
            'metric': 'auc' if CONFIG["METRIC"]=='roc_auc' else ('logloss' if CONFIG["TASK"]=='binary' else 'rmse'),
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': CONFIG["N_ESTIMATORS"],
            'random_state': CONFIG["SEED"] + fold,
            'verbosity': -1,
            'n_jobs': -1
        }
        if CONFIG["TASK"] == "multiclass":
            lgb_params['num_class'] = len(np.unique(y))
            
        model_lgb = lgb.LGBMClassifier(**lgb_params) if CONFIG["TASK"] != "regression" else lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(CONFIG["EARLY_STOPPING_ROUNDS"], verbose=False)]
        )
        
        val_pred = model_lgb.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=='binary' else (model_lgb.predict_proba(X_val) if CONFIG["TASK"]=='multiclass' else model_lgb.predict(X_val))
        oof_lgb[val_idx] = val_pred
        test_pred = model_lgb.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=='binary' else (model_lgb.predict_proba(X_test) if CONFIG["TASK"]=='multiclass' else model_lgb.predict(X_test))
        test_lgb += test_pred / CONFIG["N_SPLITS"]
        
        score = calculate_metric(y_val, val_pred, CONFIG["METRIC"], CONFIG["TASK"])
        cv_scores_lgb.append(score)
        saved_models["lgb"].append(model_lgb)
        print(f"LGBM Fold {fold + 1} Score: {score:.5f}")

    # ----------------- 2. CATBOOST -----------------
    if CONFIG["USE_CATBOOST"]:
        cb_params = {
            'iterations': CONFIG["N_ESTIMATORS"],
            'learning_rate': 0.05,
            'random_seed': CONFIG["SEED"] + fold,
            'verbose': False,
            'early_stopping_rounds': CONFIG["EARLY_STOPPING_ROUNDS"]
        }
        model_cat = cb.CatBoostClassifier(**cb_params) if CONFIG["TASK"] != "regression" else cb.CatBoostRegressor(**cb_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        val_pred_cat = model_cat.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=='binary' else (model_cat.predict_proba(X_val) if CONFIG["TASK"]=='multiclass' else model_cat.predict(X_val))
        oof_cat[val_idx] = val_pred_cat
        test_pred_cat = model_cat.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=='binary' else (model_cat.predict_proba(X_test) if CONFIG["TASK"]=='multiclass' else model_cat.predict(X_test))
        test_cat += test_pred_cat / CONFIG["N_SPLITS"]
        
        score_cat = calculate_metric(y_val, val_pred_cat, CONFIG["METRIC"], CONFIG["TASK"])
        cv_scores_cat.append(score_cat)
        saved_models["cat"].append(model_cat)
        print(f"CatBoost Fold {fold + 1} Score: {score_cat:.5f}")

    # ----------------- 3. XGBOOST -----------------
    if CONFIG["USE_XGB"]:
        xgb_params = {
            'n_estimators': CONFIG["N_ESTIMATORS"],
            'learning_rate': 0.05,
            'random_state': CONFIG["SEED"] + fold,
            'early_stopping_rounds': CONFIG["EARLY_STOPPING_ROUNDS"],
            'n_jobs': -1,
            'eval_metric': 'auc' if CONFIG["METRIC"]=='roc_auc' else 'logloss'
        }
        model_xgb = xgb.XGBClassifier(**xgb_params) if CONFIG["TASK"] != "regression" else xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=='binary' else (model_xgb.predict_proba(X_val) if CONFIG["TASK"]=='multiclass' else model_xgb.predict(X_val))
        oof_xgb[val_idx] = val_pred_xgb
        test_pred_xgb = model_xgb.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=='binary' else (model_xgb.predict_proba(X_test) if CONFIG["TASK"]=='multiclass' else model_xgb.predict(X_test))
        test_xgb += test_pred_xgb / CONFIG["N_SPLITS"]
        
        score_xgb = calculate_metric(y_val, val_pred_xgb, CONFIG["METRIC"], CONFIG["TASK"])
        cv_scores_xgb.append(score_xgb)
        saved_models["xgb"].append(model_xgb)
        print(f"XGBoost Fold {fold + 1} Score: {score_xgb:.5f}")

print(f"\n[TIME] Pelatihan selesai dalam {(time.time() - start_time):.1f} detik.")

# %%
# ==========================================
# 8. OUT-OF-FOLD EVALUATION & BLENDING
# ==========================================
print("[EVAL] --- HASIL EVALUASI OUT-OF-FOLD (OOF) KESELURUHAN ---")

score_lgb_tot = calculate_metric(y, oof_lgb, CONFIG["METRIC"], CONFIG["TASK"]) if CONFIG["USE_LGBM"] else 0
score_cat_tot = calculate_metric(y, oof_cat, CONFIG["METRIC"], CONFIG["TASK"]) if CONFIG["USE_CATBOOST"] else 0
score_xgb_tot = calculate_metric(y, oof_xgb, CONFIG["METRIC"], CONFIG["TASK"]) if CONFIG["USE_XGB"] else 0

print(f"OOF LightGBM Score: {score_lgb_tot:.5f} (Mean Folds: {np.mean(cv_scores_lgb):.5f} ± {np.std(cv_scores_lgb):.5f})")
print(f"OOF CatBoost Score: {score_cat_tot:.5f} (Mean Folds: {np.mean(cv_scores_cat):.5f} ± {np.std(cv_scores_cat):.5f})")
print(f"OOF XGBoost  Score: {score_xgb_tot:.5f} (Mean Folds: {np.mean(cv_scores_xgb):.5f} ± {np.std(cv_scores_xgb):.5f})")

# Simple Robust Blend (40% LGBM + 40% CatBoost + 20% XGBoost)
oof_blend = (0.4 * oof_lgb) + (0.4 * oof_cat) + (0.2 * oof_xgb)
test_blend = (0.4 * test_lgb) + (0.4 * test_cat) + (0.2 * test_xgb)

score_blend_tot = calculate_metric(y, oof_blend, CONFIG["METRIC"], CONFIG["TASK"])
print(f"\n[ENSEMBLE] OOF TRIO ENSEMBLE BLEND SCORE: {score_blend_tot:.5f}")

# Threshold Optimization
if CONFIG["TASK"] == "binary" and CONFIG["METRIC"] in ["f1", "f1_macro"]:
    best_th, best_f1 = optimize_threshold_for_f1(y, oof_blend)
    print(f"[METRIC] Best Probability Threshold: {best_th:.2f} (OOF F1 Naik ke {best_f1:.5f})")
else:
    best_th = 0.5

# %%
# ==========================================
# 9. EXPLAINABILITY (SHAP & FEATURE IMPORTANCE)
# ==========================================
try:
    if shap is None:
        raise ImportError("SHAP library not found")
    sample_X = X.iloc[:300]
    explainer = shap.TreeExplainer(saved_models["lgb"][0])
    shap_values = explainer.shap_values(sample_X)
    
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, sample_X, show=False, max_display=12)
    plt.title("SHAP Feature Impact & Directionality (Explainability)", fontsize=14, weight='bold')
    plt.tight_layout()
    shap_plot_path = os.path.join(CONFIG["EXPORT_DIR"], "05_shap_feature_importance.png")
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] SHAP Beeswarm tersimpan untuk slide: {shap_plot_path}")
except Exception as e:
    print(f"[INFO] Menggunakan Native Feature Importance ({e})...")
    lgb_model = saved_models["lgb"][0]
    importances = lgb_model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title("Top 15 Feature Importances (LightGBM Gain/Split)", fontsize=14, weight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    shap_plot_path = os.path.join(CONFIG["EXPORT_DIR"], "05_shap_feature_importance.png")
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Feature Importance tersimpan untuk slide: {shap_plot_path}")

# %%
# ==========================================
# 10. ERROR ANALYSIS & PERFORMANCE SCORECARD
# ==========================================
# 1. Confusion Matrix / Error Analysis Plot (Slide Bahan 4)
plt.figure(figsize=(6, 5))
if CONFIG["TASK"] == "binary":
    y_pred_binary = (oof_blend >= best_th).astype(int)
    cm = confusion_matrix(y, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.title(f"Confusion Matrix (OOF Ensemble)\nThreshold = {best_th:.2f}", fontsize=13, weight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
elif CONFIG["TASK"] == "regression":
    sns.scatterplot(x=y, y=oof_blend, alpha=0.5, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Residuals (OOF)", fontsize=13, weight='bold')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")

plt.tight_layout()
eval_plot_path = os.path.join(CONFIG["EXPORT_DIR"], "04_model_evaluation_matrix.png")
plt.savefig(eval_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan untuk slide presentasi: {eval_plot_path}")

# 2. Performance Comparison Scorecard (Slide Bahan 5)
models_summary = pd.DataFrame({
    'Model': ['LightGBM', 'CatBoost', 'XGBoost', 'Trio Ensemble'],
    'OOF_Score': [score_lgb_tot, score_cat_tot, score_xgb_tot, score_blend_tot],
    'Std_Folds': [np.std(cv_scores_lgb), np.std(cv_scores_cat), np.std(cv_scores_xgb), 0.0]
})

plt.figure(figsize=(7, 4))
bars = plt.barh(models_summary['Model'], models_summary['OOF_Score'], color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c'])
plt.xlim(min(models_summary['OOF_Score']) * 0.95, max(models_summary['OOF_Score']) * 1.02)
plt.title(f"Model Performance Scorecard ({CONFIG['METRIC'].upper()})", fontsize=14, weight='bold')
plt.xlabel("Out-Of-Fold Metric Score")

for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}',
             ha='left', va='center', fontsize=11, weight='bold')

plt.tight_layout()
scorecard_path = os.path.join(CONFIG["EXPORT_DIR"], "03_cv_score_comparison.png")
plt.savefig(scorecard_path, dpi=300)
plt.close()
print(f"[PLOT] Tersimpan untuk slide presentasi: {scorecard_path}")

# %%
# ==========================================
# 11. FINAL SUBMISSIONS & PRESENTATION ASSETS ZIP
# ==========================================
# 1. Generate Submission 1: Trio Blend (Rekomendasi Utama)
sub_blend = pd.DataFrame({
    CONFIG["ID_COL"]: df_test_fe[CONFIG["ID_COL"]]
})

if CONFIG["TASK"] == "binary":
    sub_blend[CONFIG["TARGET_COL"]] = test_blend
else:
    sub_blend[CONFIG["TARGET_COL"]] = test_blend

sub_blend.to_csv("submission_ensemble_best_cv.csv", index=False)
print("[OK] Submission 1 berhasil disimpan: submission_ensemble_best_cv.csv")

# 2. Generate Submission 2: Single Best Model (Fallback)
sub_single = pd.DataFrame({
    CONFIG["ID_COL"]: df_test_fe[CONFIG["ID_COL"]],
    CONFIG["TARGET_COL"]: test_lgb
})
sub_single.to_csv("submission_single_lgbm.csv", index=False)
print("[OK] Submission 2 berhasil disimpan: submission_single_lgbm.csv")

# 3. Sanity Check File Submission
print("\n[AUDIT] Sanity Check Submission:")
print(f"Null count: {sub_blend[CONFIG['TARGET_COL']].isnull().sum()}")
print(f"Total baris: {len(sub_blend)} (Harus sama dengan {len(df_test)})")
print(sub_blend.head())

# 4. Arsipkan seluruh gambar slide ke file .zip untuk download instan
shutil.make_archive("presentation_assets", 'zip', CONFIG["EXPORT_DIR"])
print(f"\n[ZIP] Seluruh gambar presentasi telah dibundel ke: presentation_assets.zip")
print("[READY] SIAP PRESENTASI! Buka folder aset untuk ditempel ke template slide panitia.")
