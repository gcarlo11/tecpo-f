# %%
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import os
import gc
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import rankdata

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss, accuracy_score,
    mean_squared_error, mean_absolute_error,
    confusion_matrix, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150

# %%
CONFIG = {
    "TRAIN_PATH": "train.csv",
    "TEST_PATH": "test.csv",
    "SAMPLE_SUB_PATH": "sample_submission.csv",
    "EXPORT_DIR": "./presentation_assets_grandmaster/",
    "ID_COL": "id",
    "TARGET_COL": "target",
    "TASK": "binary",
    "METRIC": "roc_auc",
    "N_SPLITS": 5,
    "SEED": 42,
    "N_CLUSTERS": 6,
    "N_PCA_COMPONENTS": 3,
    "USE_ADVERSARIAL_VALIDATION": True,
    "EARLY_STOPPING_ROUNDS": 80,
    "N_ESTIMATORS": 1200,
}

os.makedirs(CONFIG["EXPORT_DIR"], exist_ok=True)

# %%
def seed_everything(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CONFIG["SEED"])

def calculate_metric(y_true, y_pred, metric_name, task="binary"):
    if metric_name == "roc_auc":
        return roc_auc_score(y_true, y_pred) if task == "binary" else roc_auc_score(y_true, y_pred, multi_class='ovr')
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
    raise ValueError(f"Unknown metric: {metric_name}")

def optimize_threshold(y_true, y_prob):
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        score = f1_score(y_true, (y_prob >= thresh).astype(int), average='macro' if CONFIG["TASK"]=="multiclass" else 'binary')
        if score > best_f1:
            best_f1, best_thresh = score, thresh
    return best_thresh, best_f1

def rank_average(preds_list):
    ranked_preds = [rankdata(p) / len(p) for p in preds_list]
    return np.mean(ranked_preds, axis=0)

# %%
if not os.path.exists(CONFIG["TRAIN_PATH"]):
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=2500, n_features=16, n_informative=11, random_state=42)
    df_train = pd.DataFrame(X_syn, columns=[f"feat_{i}" for i in range(16)])
    df_train[CONFIG["ID_COL"]] = range(1, len(df_train) + 1)
    df_train[CONFIG["TARGET_COL"]] = y_syn
    df_train["category_A"] = np.random.choice(["Alpha", "Beta", "Gamma"], size=len(df_train))
    df_train["category_B"] = np.random.choice(["High", "Med", "Low"], size=len(df_train))
    df_train.iloc[0:25, 2] = np.nan
    df_test = df_train.copy().drop(columns=[CONFIG["TARGET_COL"]]).iloc[:600].reset_index(drop=True)
    df_test[CONFIG["ID_COL"]] = range(6000, 6000 + len(df_test))
else:
    df_train = pd.read_csv(CONFIG["TRAIN_PATH"])
    df_test = pd.read_csv(CONFIG["TEST_PATH"])

print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

# %%
features_raw = [c for c in df_train.columns if c not in [CONFIG["ID_COL"], CONFIG["TARGET_COL"]]]
av_train = df_train[features_raw].copy()
av_test = df_test[features_raw].copy()
av_train['is_test'] = 0
av_test['is_test'] = 1
av_combined = pd.concat([av_train, av_test], axis=0).reset_index(drop=True)

av_cat = [c for c in features_raw if av_combined[c].dtype == 'object' or av_combined[c].dtype.name == 'category']
for c in av_cat:
    le = LabelEncoder()
    av_combined[c] = le.fit_transform(av_combined[c].astype(str))

av_X = av_combined[features_raw]
av_y = av_combined['is_test']

av_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG["SEED"])
av_oof = np.zeros(len(av_combined))

for tr_idx, val_idx in av_kf.split(av_X, av_y):
    m_av = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, random_state=CONFIG["SEED"], verbosity=-1, n_jobs=-1)
    m_av.fit(av_X.iloc[tr_idx], av_y.iloc[tr_idx])
    av_oof[val_idx] = m_av.predict_proba(av_X.iloc[val_idx])[:, 1]

av_auc = roc_auc_score(av_y, av_oof)
print(f"Adversarial Validation AUC: {av_auc:.4f}")
if av_auc > 0.70:
    print("WARNING: Train and Test distributions differ (Covariate Shift detected).")
else:
    print("STATUS: Train and Test distributions match well (No significant shift).")

fpr, tpr, _ = roc_curve(av_y, av_oof)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AV AUC = {av_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Adversarial Validation (Train vs Test Shift)", fontsize=13, weight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "01_adversarial_validation_drift.png"), dpi=300)
plt.close()

# %%
def grandmaster_feature_factory(train_data, test_data, config):
    df_tr = train_data.copy()
    df_te = test_data.copy()
    target_col = config["TARGET_COL"]
    id_col = config["ID_COL"]

    features = [c for c in df_tr.columns if c not in [id_col, target_col]]
    cat_cols = [c for c in features if df_tr[c].dtype == 'object' or df_tr[c].dtype.name == 'category']
    num_cols = [c for c in features if c not in cat_cols]

    # 1. Row-level statistics
    for df in [df_tr, df_te]:
        df['row_null_count'] = df[num_cols].isnull().sum(axis=1)
        df['row_mean'] = df[num_cols].mean(axis=1)
        df['row_std'] = df[num_cols].std(axis=1).fillna(0)
        df['row_min'] = df[num_cols].min(axis=1)
        df['row_max'] = df[num_cols].max(axis=1)

    # 2. Categorical Frequency & Interactions
    for c in cat_cols:
        freq = df_tr[c].value_counts().to_dict()
        df_tr[f"{c}_freq"] = df_tr[c].map(freq).fillna(0)
        df_te[f"{c}_freq"] = df_te[c].map(freq).fillna(0)

    if len(cat_cols) >= 2:
        df_tr['cat_interact'] = df_tr[cat_cols[0]].astype(str) + "_" + df_tr[cat_cols[1]].astype(str)
        df_te['cat_interact'] = df_te[cat_cols[0]].astype(str) + "_" + df_te[cat_cols[1]].astype(str)
        inter_freq = df_tr['cat_interact'].value_counts().to_dict()
        df_tr['cat_interact_freq'] = df_tr['cat_interact'].map(inter_freq).fillna(0)
        df_te['cat_interact_freq'] = df_te['cat_interact'].map(inter_freq).fillna(0)
        cat_cols.append('cat_interact')

    # 3. Group Aggregations
    if len(cat_cols) > 0 and len(num_cols) > 0:
        main_cat = cat_cols[0]
        for nc in num_cols[:3]:
            agg = df_tr.groupby(main_cat)[nc].agg(['mean', 'std']).reset_index()
            agg.columns = [main_cat, f"{main_cat}_{nc}_grp_mean", f"{main_cat}_{nc}_grp_std"]
            df_tr = df_tr.merge(agg, on=main_cat, how='left')
            df_te = df_te.merge(agg, on=main_cat, how='left')

    # 4. Label Encoding
    for c in cat_cols:
        le = LabelEncoder()
        all_vals = list(df_tr[c].astype(str).values) + list(df_te[c].astype(str).values)
        le.fit(all_vals)
        df_tr[c] = le.transform(df_tr[c].astype(str))
        df_te[c] = le.transform(df_te[c].astype(str))

    # 5. K-Means Cluster Distance & PCA Projections
    scaler = StandardScaler()
    num_imputed_tr = df_tr[num_cols].fillna(df_tr[num_cols].median())
    num_imputed_te = df_te[num_cols].fillna(df_tr[num_cols].median())
    
    scaled_tr = scaler.fit_transform(num_imputed_tr)
    scaled_te = scaler.transform(num_imputed_te)

    pca = PCA(n_components=config["N_PCA_COMPONENTS"], random_state=config["SEED"])
    pca_tr = pca.fit_transform(scaled_tr)
    pca_te = pca.transform(scaled_te)
    for i in range(config["N_PCA_COMPONENTS"]):
        df_tr[f"pca_{i}"] = pca_tr[:, i]
        df_te[f"pca_{i}"] = pca_te[:, i]

    kmeans = KMeans(n_clusters=config["N_CLUSTERS"], random_state=config["SEED"], n_init=10)
    df_tr['cluster_id'] = kmeans.fit_predict(scaled_tr)
    df_te['cluster_id'] = kmeans.predict(scaled_te)

    # Distances to cluster centroids
    dist_tr = kmeans.transform(scaled_tr)
    dist_te = kmeans.transform(scaled_te)
    for i in range(config["N_CLUSTERS"]):
        df_tr[f"dist_cluster_{i}"] = dist_tr[:, i]
        df_te[f"dist_cluster_{i}"] = dist_te[:, i]

    return df_tr, df_te

df_train_fe, df_test_fe = grandmaster_feature_factory(df_train, df_test, CONFIG)
print(f"Post-FE Dimensions: Train {df_train_fe.shape}, Test {df_test_fe.shape}")

# Visual Cluster Plot
plt.figure(figsize=(7, 5))
scatter = plt.scatter(df_train_fe['pca_0'], df_train_fe['pca_1'], c=df_train_fe['cluster_id'], cmap='tab10', alpha=0.5, s=20)
plt.title("Unsupervised Cluster & PCA Space", fontsize=13, weight='bold')
plt.xlabel("PCA Component 0")
plt.ylabel("PCA Component 1")
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "02_feature_space_clusters.png"), dpi=300)
plt.close()

# %%
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

# Level-1 Base Learners: LightGBM, CatBoost, XGBoost, and Regularized Linear
base_models = ["lgb", "cat", "xgb", "linear"]
oof_matrix = {m: np.zeros(len(df_train_fe)) for m in base_models}
test_matrix = {m: np.zeros(len(df_test_fe)) for m in base_models}
cv_scores = {m: [] for m in base_models}
fitted_models = {m: [] for m in base_models}

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(split_gen):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Model 1: LightGBM (Leaf-wise GBDT)
    lgb_params = {
        'objective': 'binary' if CONFIG["TASK"]=='binary' else 'regression',
        'metric': 'auc' if CONFIG["METRIC"]=='roc_auc' else 'logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'n_estimators': CONFIG["N_ESTIMATORS"],
        'random_state': CONFIG["SEED"] + fold,
        'verbosity': -1,
        'n_jobs': -1
    }
    m_lgb = lgb.LGBMClassifier(**lgb_params) if CONFIG["TASK"]!="regression" else lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(CONFIG["EARLY_STOPPING_ROUNDS"], verbose=False)])
    pred_val_lgb = m_lgb.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=="binary" else m_lgb.predict(X_val)
    oof_matrix["lgb"][val_idx] = pred_val_lgb
    pred_test_lgb = m_lgb.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=="binary" else m_lgb.predict(X_test)
    test_matrix["lgb"] += pred_test_lgb / CONFIG["N_SPLITS"]
    cv_scores["lgb"].append(calculate_metric(y_val, pred_val_lgb, CONFIG["METRIC"], CONFIG["TASK"]))
    fitted_models["lgb"].append(m_lgb)

    # Model 2: CatBoost (Symmetric Oblivious Trees)
    cb_params = {
        'iterations': CONFIG["N_ESTIMATORS"],
        'learning_rate': 0.05,
        'random_seed': CONFIG["SEED"] + fold,
        'verbose': False,
        'early_stopping_rounds': CONFIG["EARLY_STOPPING_ROUNDS"]
    }
    m_cat = cb.CatBoostClassifier(**cb_params) if CONFIG["TASK"]!="regression" else cb.CatBoostRegressor(**cb_params)
    m_cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    pred_val_cat = m_cat.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=="binary" else m_cat.predict(X_val)
    oof_matrix["cat"][val_idx] = pred_val_cat
    pred_test_cat = m_cat.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=="binary" else m_cat.predict(X_test)
    test_matrix["cat"] += pred_test_cat / CONFIG["N_SPLITS"]
    cv_scores["cat"].append(calculate_metric(y_val, pred_val_cat, CONFIG["METRIC"], CONFIG["TASK"]))
    fitted_models["cat"].append(m_cat)

    # Model 3: XGBoost (Depth-wise GBDT)
    xgb_params = {
        'n_estimators': CONFIG["N_ESTIMATORS"],
        'learning_rate': 0.05,
        'random_state': CONFIG["SEED"] + fold,
        'early_stopping_rounds': CONFIG["EARLY_STOPPING_ROUNDS"],
        'n_jobs': -1,
        'eval_metric': 'auc' if CONFIG["METRIC"]=='roc_auc' else 'logloss'
    }
    m_xgb = xgb.XGBClassifier(**xgb_params) if CONFIG["TASK"]!="regression" else xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_val_xgb = m_xgb.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=="binary" else m_xgb.predict(X_val)
    oof_matrix["xgb"][val_idx] = pred_val_xgb
    pred_test_xgb = m_xgb.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=="binary" else m_xgb.predict(X_test)
    test_matrix["xgb"] += pred_test_xgb / CONFIG["N_SPLITS"]
    cv_scores["xgb"].append(calculate_metric(y_val, pred_val_xgb, CONFIG["METRIC"], CONFIG["TASK"]))
    fitted_models["xgb"].append(m_xgb)

    # Model 4: Regularized Linear Model (Diversity against tree bias)
    med_val = X_train.median()
    sc_lin = StandardScaler()
    X_tr_sc = sc_lin.fit_transform(X_train.fillna(med_val))
    X_val_sc = sc_lin.transform(X_val.fillna(med_val))
    X_te_sc = sc_lin.transform(X_test.fillna(med_val))
    
    m_lin = LogisticRegression(C=0.1, max_iter=500, random_state=CONFIG["SEED"]) if CONFIG["TASK"]=="binary" else Ridge(alpha=1.0)
    m_lin.fit(X_tr_sc, y_train)
    pred_val_lin = m_lin.predict_proba(X_val_sc)[:, 1] if CONFIG["TASK"]=="binary" else m_lin.predict(X_val_sc)
    oof_matrix["linear"][val_idx] = pred_val_lin
    pred_test_lin = m_lin.predict_proba(X_te_sc)[:, 1] if CONFIG["TASK"]=="binary" else m_lin.predict(X_te_sc)
    test_matrix["linear"] += pred_test_lin / CONFIG["N_SPLITS"]
    cv_scores["linear"].append(calculate_metric(y_val, pred_val_lin, CONFIG["METRIC"], CONFIG["TASK"]))
    fitted_models["linear"].append(m_lin)

print(f"Level-1 Models trained in {(time.time() - start_time):.1f}s")
for m in base_models:
    score = calculate_metric(y, oof_matrix[m], CONFIG["METRIC"], CONFIG["TASK"])
    print(f"  Level-1 {m.upper():<7}: OOF {score:.5f} (CV: {np.mean(cv_scores[m]):.5f} +/- {np.std(cv_scores[m]):.5f})")

# %%
S_train = np.column_stack([oof_matrix[m] for m in base_models])
S_test = np.column_stack([test_matrix[m] for m in base_models])

# Level-2 Stacking Meta-Learner (Regularized Linear Model)
if CONFIG["TASK"] == "binary":
    meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=CONFIG["SEED"])
    meta_learner.fit(S_train, y)
    oof_stacked = meta_learner.predict_proba(S_train)[:, 1]
    test_stacked = meta_learner.predict_proba(S_test)[:, 1]
    stack_weights = meta_learner.coef_[0]
elif CONFIG["TASK"] == "multiclass":
    meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=CONFIG["SEED"])
    meta_learner.fit(S_train, y)
    oof_stacked = meta_learner.predict_proba(S_train)
    test_stacked = meta_learner.predict_proba(S_test)
    stack_weights = np.mean(meta_learner.coef_, axis=0)
else:
    meta_learner = Ridge(alpha=1.0, random_state=CONFIG["SEED"])
    meta_learner.fit(S_train, y)
    oof_stacked = meta_learner.predict(S_train)
    test_stacked = meta_learner.predict(S_test)
    stack_weights = meta_learner.coef_

oof_rank_blend = rank_average([oof_matrix[m] for m in base_models])
test_rank_blend = rank_average([test_matrix[m] for m in base_models])

score_stacked = calculate_metric(y, oof_stacked, CONFIG["METRIC"], CONFIG["TASK"])
score_rank = calculate_metric(y, oof_rank_blend, CONFIG["METRIC"], CONFIG["TASK"])

print("\n--- STACKING ENSEMBLE RESULTS ---")
for m, w in zip(base_models, stack_weights):
    print(f"  Meta-Weight {m.upper():<7}: {w:.4f}")
print(f"Level-2 Stacked Meta-Learner OOF: {score_stacked:.5f}")
print(f"Rank-Averaged Ensemble OOF     : {score_rank:.5f}")

if CONFIG["TASK"] == "binary" and CONFIG["METRIC"] in ["f1", "f1_macro"]:
    best_th, best_f1 = optimize_threshold(y, oof_stacked)
    print(f"Optimal Decision Threshold: {best_th:.2f} (F1 Score: {best_f1:.5f})")
else:
    best_th = 0.5

# Visual Comparison Chart
models_eval = [m.upper() for m in base_models] + ['RANK BLEND', 'LEVEL-2 STACK']
scores_eval = [calculate_metric(y, oof_matrix[m], CONFIG["METRIC"], CONFIG["TASK"]) for m in base_models] + [score_rank, score_stacked]

plt.figure(figsize=(8, 4))
bars = plt.barh(models_eval, scores_eval, color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#f39c12', '#e74c3c'])
plt.xlim(min(scores_eval) * 0.95, max(scores_eval) * 1.01)
plt.title(f"Level-1 vs Level-2 Stacking Performance ({CONFIG['METRIC'].upper()})", fontsize=13, weight='bold')
plt.xlabel("Out-Of-Fold Metric Score")
for bar in bars:
    w_val = bar.get_width()
    plt.text(w_val, bar.get_y() + bar.get_height()/2, f' {w_val:.5f}', ha='left', va='center', fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "03_stacking_weights_and_gains.png"), dpi=300)
plt.close()

# %%
try:
    if shap is None:
        raise ImportError("SHAP not available")
    sample_X = X.iloc[:300]
    explainer = shap.TreeExplainer(fitted_models["lgb"][0])
    shap_values = explainer.shap_values(sample_X)
    shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, sample_X, show=False, max_display=12)
    plt.title("Grandmaster Feature Impact (SHAP Beeswarm)", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "05_shap_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()
except Exception:
    m_lgb = fitted_models["lgb"][0]
    feat_imp = pd.Series(m_lgb.feature_importances_, index=features).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title("Top 15 Feature Importances", fontsize=14, weight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "05_shap_beeswarm.png"), dpi=300)
    plt.close()

# %%
plt.figure(figsize=(6, 5))
if CONFIG["TASK"] == "binary":
    y_pred_binary = (oof_stacked >= best_th).astype(int)
    cm = confusion_matrix(y, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.title(f"Stacked Ensemble Confusion Matrix\nThreshold = {best_th:.2f}", fontsize=13, weight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
elif CONFIG["TASK"] == "regression":
    sns.scatterplot(x=y, y=oof_stacked, alpha=0.5, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Stacked Ensemble Residuals", fontsize=13, weight='bold')
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "04_confusion_matrix_stacked.png"), dpi=300)
plt.close()

# %%
# Submission 1: Level-2 Stacking (Primary Winning Submission)
sub_stack = pd.DataFrame({
    CONFIG["ID_COL"]: df_test_fe[CONFIG["ID_COL"]],
    CONFIG["TARGET_COL"]: test_stacked
})
sub_stack.to_csv("submission_grandmaster_stacking.csv", index=False)

# Submission 2: Rank-Averaged Ensemble (Calibration-Independent Alternative)
sub_rank = pd.DataFrame({
    CONFIG["ID_COL"]: df_test_fe[CONFIG["ID_COL"]],
    CONFIG["TARGET_COL"]: test_rank_blend
})
sub_rank.to_csv("submission_rank_average.csv", index=False)

shutil.make_archive("presentation_assets_grandmaster", 'zip', CONFIG["EXPORT_DIR"])

print(f"Stacking Submission: {len(sub_stack)} rows, {sub_stack[CONFIG['TARGET_COL']].isnull().sum()} nulls.")
print(f"Rank-Average Submission: {len(sub_rank)} rows, {sub_rank[CONFIG['TARGET_COL']].isnull().sum()} nulls.")
print("Assets archived to: presentation_assets_grandmaster.zip")

