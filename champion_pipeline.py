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

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss, accuracy_score,
    mean_squared_error, mean_absolute_error,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

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
    "EXPORT_DIR": "./presentation_assets/",
    "ID_COL": "id",
    "TARGET_COL": "target",
    "TASK": "binary",
    "METRIC": "roc_auc",
    "N_SPLITS": 5,
    "SEED": 42,
    "USE_LGBM": True,
    "USE_CATBOOST": True,
    "USE_XGB": True,
    "EARLY_STOPPING_ROUNDS": 100,
    "N_ESTIMATORS": 1500,
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

def optimize_weights(oof_preds_dict, y_true):
    model_keys = list(oof_preds_dict.keys())
    preds_matrix = np.column_stack([oof_preds_dict[k] for k in model_keys])
    n_models = len(model_keys)

    def objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blend = np.dot(preds_matrix, w)
        score = calculate_metric(y_true, blend, CONFIG["METRIC"], CONFIG["TASK"])
        if CONFIG["METRIC"] in ["roc_auc", "f1", "f1_macro", "accuracy"]:
            return -score
        return score

    init_w = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_w = res.x / np.sum(res.x)
    return dict(zip(model_keys, opt_w))

# %%
if not os.path.exists(CONFIG["TRAIN_PATH"]):
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

print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

# %%
plt.figure(figsize=(7, 4))
if CONFIG["TASK"] in ["binary", "multiclass"]:
    ax = sns.countplot(data=df_train, x=CONFIG["TARGET_COL"], palette="mako")
    total = len(df_train)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}\n({height/total:.1%})',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', weight='bold')
    plt.title("Target Distribution", fontsize=14, weight='bold')
else:
    sns.histplot(df_train[CONFIG["TARGET_COL"]], kde=True, color='teal')
    plt.title("Target Distribution", fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "01_eda_target_distribution.png"), dpi=300)
plt.close()

null_counts = df_train.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if len(null_counts) > 0:
    plt.figure(figsize=(8, 4))
    sns.barplot(x=null_counts.values, y=null_counts.index, palette="flare")
    plt.title("Missing Values Audit", fontsize=14, weight='bold')
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "02_missing_values_matrix.png"), dpi=300)
    plt.close()

# %%
def apply_feature_engineering(train_data, test_data, config):
    df_tr = train_data.copy()
    df_te = test_data.copy()
    target_col = config["TARGET_COL"]
    id_col = config["ID_COL"]

    features = [c for c in df_tr.columns if c not in [id_col, target_col]]
    cat_cols = [c for c in features if df_tr[c].dtype == 'object' or df_tr[c].dtype.name == 'category']
    num_cols = [c for c in features if c not in cat_cols]

    for c in cat_cols:
        freq = df_tr[c].value_counts().to_dict()
        df_tr[f"{c}_freq"] = df_tr[c].map(freq).fillna(0)
        df_te[f"{c}_freq"] = df_te[c].map(freq).fillna(0)

    if len(cat_cols) > 0 and len(num_cols) > 0:
        main_cat = cat_cols[0]
        for num_c in num_cols[:3]:
            agg_stats = df_tr.groupby(main_cat)[num_c].agg(['mean', 'std']).reset_index()
            agg_stats.columns = [main_cat, f"{main_cat}_{num_c}_grp_mean", f"{main_cat}_{num_c}_grp_std"]
            df_tr = df_tr.merge(agg_stats, on=main_cat, how='left')
            df_te = df_te.merge(agg_stats, on=main_cat, how='left')

    for c in cat_cols:
        le = LabelEncoder()
        all_vals = list(df_tr[c].astype(str).values) + list(df_te[c].astype(str).values)
        le.fit(all_vals)
        df_tr[c] = le.transform(df_tr[c].astype(str))
        df_te[c] = le.transform(df_te[c].astype(str))

    return df_tr, df_te

df_train_fe, df_test_fe = apply_feature_engineering(df_train, df_test, CONFIG)
print(f"Post-FE shapes: Train {df_train_fe.shape}, Test {df_test_fe.shape}")

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

oof_lgb = np.zeros(len(df_train_fe)) if CONFIG["TASK"] == "binary" else np.zeros((len(df_train_fe), len(np.unique(y))))
test_lgb = np.zeros(len(df_test_fe)) if CONFIG["TASK"] == "binary" else np.zeros((len(df_test_fe), len(np.unique(y))))

oof_cat = np.zeros_like(oof_lgb)
test_cat = np.zeros_like(test_lgb)

oof_xgb = np.zeros_like(oof_lgb)
test_xgb = np.zeros_like(test_lgb)

cv_scores_lgb, cv_scores_cat, cv_scores_xgb = [], [], []
saved_models = {"lgb": [], "cat": [], "xgb": []}

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(split_gen):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

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
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(CONFIG["EARLY_STOPPING_ROUNDS"], verbose=False)])

        val_pred = model_lgb.predict_proba(X_val)[:, 1] if CONFIG["TASK"]=='binary' else (model_lgb.predict_proba(X_val) if CONFIG["TASK"]=='multiclass' else model_lgb.predict(X_val))
        oof_lgb[val_idx] = val_pred
        test_pred = model_lgb.predict_proba(X_test)[:, 1] if CONFIG["TASK"]=='binary' else (model_lgb.predict_proba(X_test) if CONFIG["TASK"]=='multiclass' else model_lgb.predict(X_test))
        test_lgb += test_pred / CONFIG["N_SPLITS"]
        cv_scores_lgb.append(calculate_metric(y_val, val_pred, CONFIG["METRIC"], CONFIG["TASK"]))
        saved_models["lgb"].append(model_lgb)

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
        cv_scores_cat.append(calculate_metric(y_val, val_pred_cat, CONFIG["METRIC"], CONFIG["TASK"]))
        saved_models["cat"].append(model_cat)

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
        cv_scores_xgb.append(calculate_metric(y_val, val_pred_xgb, CONFIG["METRIC"], CONFIG["TASK"]))
        saved_models["xgb"].append(model_xgb)

print(f"Training completed in {(time.time() - start_time):.1f}s")

# %%
score_lgb_tot = calculate_metric(y, oof_lgb, CONFIG["METRIC"], CONFIG["TASK"]) if CONFIG["USE_LGBM"] else 0
score_cat_tot = calculate_metric(y, oof_cat, CONFIG["METRIC"], CONFIG["TASK"]) if CONFIG["USE_CATBOOST"] else 0
score_xgb_tot = calculate_metric(y, oof_xgb, CONFIG["METRIC"], CONFIG["TASK"]) if CONFIG["USE_XGB"] else 0

print(f"OOF LightGBM: {score_lgb_tot:.5f} (CV: {np.mean(cv_scores_lgb):.5f} +/- {np.std(cv_scores_lgb):.5f})")
print(f"OOF CatBoost: {score_cat_tot:.5f} (CV: {np.mean(cv_scores_cat):.5f} +/- {np.std(cv_scores_cat):.5f})")
print(f"OOF XGBoost:  {score_xgb_tot:.5f} (CV: {np.mean(cv_scores_xgb):.5f} +/- {np.std(cv_scores_xgb):.5f})")

oof_dict = {}
test_dict = {}
if CONFIG["USE_LGBM"]:
    oof_dict["lgb"] = oof_lgb
    test_dict["lgb"] = test_lgb
if CONFIG["USE_CATBOOST"]:
    oof_dict["cat"] = oof_cat
    test_dict["cat"] = test_cat
if CONFIG["USE_XGB"]:
    oof_dict["xgb"] = oof_xgb
    test_dict["xgb"] = test_xgb

opt_weights = optimize_weights(oof_dict, y)
print("Optimal Weights:")
for k, w in opt_weights.items():
    print(f"  {k}: {w:.4f}")

oof_blend = sum(opt_weights[k] * oof_dict[k] for k in opt_weights)
test_blend = sum(opt_weights[k] * test_dict[k] for k in opt_weights)

score_blend_tot = calculate_metric(y, oof_blend, CONFIG["METRIC"], CONFIG["TASK"])
print(f"Optimal OOF Ensemble Score: {score_blend_tot:.5f}")

if CONFIG["TASK"] == "binary" and CONFIG["METRIC"] in ["f1", "f1_macro"]:
    best_th, best_f1 = optimize_threshold(y, oof_blend)
    print(f"Optimal Threshold: {best_th:.2f} (F1: {best_f1:.5f})")
else:
    best_th = 0.5

# %%
try:
    if shap is None:
        raise ImportError("SHAP not available")
    sample_X = X.iloc[:300]
    explainer = shap.TreeExplainer(saved_models["lgb"][0])
    shap_values = explainer.shap_values(sample_X)
    shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, sample_X, show=False, max_display=12)
    plt.title("Feature Impact (SHAP Beeswarm)", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "05_shap_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
except Exception:
    lgb_model = saved_models["lgb"][0]
    feat_imp = pd.Series(lgb_model.feature_importances_, index=features).sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title("Top 15 Feature Importances", fontsize=14, weight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "05_shap_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

# %%
plt.figure(figsize=(6, 5))
if CONFIG["TASK"] == "binary":
    y_pred_binary = (oof_blend >= best_th).astype(int)
    cm = confusion_matrix(y, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.title(f"Confusion Matrix (Threshold = {best_th:.2f})", fontsize=13, weight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
elif CONFIG["TASK"] == "regression":
    sns.scatterplot(x=y, y=oof_blend, alpha=0.5, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted Residuals", fontsize=13, weight='bold')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "04_model_evaluation_matrix.png"), dpi=300)
plt.close()

models_summary = pd.DataFrame({
    'Model': ['LightGBM', 'CatBoost', 'XGBoost', 'Optimal Ensemble'],
    'Score': [score_lgb_tot, score_cat_tot, score_xgb_tot, score_blend_tot]
})

plt.figure(figsize=(7, 4))
bars = plt.barh(models_summary['Model'], models_summary['Score'], color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c'])
plt.xlim(min(models_summary['Score']) * 0.95, max(models_summary['Score']) * 1.02)
plt.title(f"Model Performance Scorecard ({CONFIG['METRIC'].upper()})", fontsize=14, weight='bold')
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f' {width:.4f}', ha='left', va='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["EXPORT_DIR"], "03_cv_score_comparison.png"), dpi=300)
plt.close()

# %%
sub_blend = pd.DataFrame({
    CONFIG["ID_COL"]: df_test_fe[CONFIG["ID_COL"]],
    CONFIG["TARGET_COL"]: test_blend
})
sub_blend.to_csv("submission_ensemble_best_cv.csv", index=False)

sub_single = pd.DataFrame({
    CONFIG["ID_COL"]: df_test_fe[CONFIG["ID_COL"]],
    CONFIG["TARGET_COL"]: test_lgb
})
sub_single.to_csv("submission_single_lgbm.csv", index=False)

shutil.make_archive("presentation_assets", 'zip', CONFIG["EXPORT_DIR"])

print(f"Submissions saved: {len(sub_blend)} rows. Nulls: {sub_blend[CONFIG['TARGET_COL']].isnull().sum()}")
print("Presentation assets archived to presentation_assets.zip")

