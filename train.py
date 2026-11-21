"""
train.py
--------
dataset -> preprocessing -> train/test split -> train LR / DT / RF
-> evaluate & compare -> select best -> save.

Evaluation:
  * Repeated stratified 5-fold cross-validation (primary)
  * Classic 80/20 hold-out split (as in the synopsis)
  * LEAKAGE EXPERIMENT: same models WITH the 'Health_condition' column, to show
    why it is excluded from the final model.

Run:  python train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config as C

RS = 42
BASE = "Baseline (always 'Lower')"

df = pd.read_csv(C.DATA_PATH)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
y = df[C.TARGET]
print("Class balance:", y.value_counts().to_dict())

num_cols = C.NUMERIC_FEATURES
cat_cols = list(C.CATEGORICAL_FEATURES)
X = df[num_cols + cat_cols]

# ------------------------------------------------------------ EDA plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
y.map({0: "Lower Risk", 1: "Higher Risk"}).value_counts().plot(
    kind="bar", ax=ax[0], color=["#4c78a8", "#e45756"])
ax[0].set_title("Class distribution"); ax[0].tick_params(axis="x", rotation=0)
df.groupby("Family_background")[C.TARGET].mean().plot(kind="bar", ax=ax[1], color="#72b7b2")
ax[1].set_title("Higher-risk rate by family background"); ax[1].set_ylim(0, 1)
ax[1].tick_params(axis="x", rotation=15)
df.groupby("Health_condition")[C.TARGET].mean().reindex(
    ["Good", "Normal", "Abnormal", "Critical"]).plot(kind="bar", ax=ax[2], color="#f58518")
ax[2].set_title("Higher-risk rate by Health_condition\n(label proxy -> excluded)")
ax[2].set_ylim(0, 1); ax[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("reports/eda.png", dpi=130); plt.close()


def make_pipe(clf, nums, cats):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def models():
    return {
        BASE: DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                class_weight="balanced", random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=10,
                                                class_weight="balanced", random_state=RS, n_jobs=-1),
    }


scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)

# ------------------------------------------------------------ main CV
cv_mean, cv_std, cms = {}, {}, {}
for name, clf in models().items():
    s = cross_validate(make_pipe(clf, num_cols, cat_cols), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_mean[name] = {m: s[f"test_{m}"].mean() for m in scoring}
    cv_std[name] = {m: s[f"test_{m}"].std() for m in scoring}
    pred = cross_val_predict(make_pipe(clf, num_cols, cat_cols), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=RS))
    cms[name] = confusion_matrix(y, pred)
cv_df = pd.DataFrame(cv_mean).T.round(3)
print("\n=== Repeated 5-fold CV (mean), WITHOUT Health_condition ===")
print(cv_df.to_string())

# ------------------------------------------------------------ hold-out
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
hold = {}
for name, clf in models().items():
    m = make_pipe(clf, num_cols, cat_cols).fit(X_tr, y_tr)
    p = m.predict(X_te)
    hold[name] = {"accuracy": accuracy_score(y_te, p),
                  "precision": precision_score(y_te, p, zero_division=0),
                  "recall": recall_score(y_te, p), "f1": f1_score(y_te, p),
                  "roc_auc": roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])}
hold_df = pd.DataFrame(hold).T.round(3)
print(f"\n=== 80/20 hold-out (train={len(X_tr)}, test={len(X_te)}) ===")
print(hold_df.to_string())

# ------------------------------------------------------------ leakage experiment
X_leak = df[num_cols + cat_cols + [C.LEAKY]]
leak = {}
for name, clf in models().items():
    if name == BASE:
        continue
    s = cross_validate(make_pipe(clf, num_cols, cat_cols + [C.LEAKY]), X_leak, y,
                       cv=cv, scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"}, n_jobs=-1)
    leak[name] = {"accuracy": s["test_accuracy"].mean(), "roc_auc": s["test_roc_auc"].mean(),
                  "roc_auc_without": cv_mean[name]["roc_auc"]}
leak_df = pd.DataFrame(leak).T.round(3)
print("\n=== Leakage experiment: WITH Health_condition ===")
print(leak_df.to_string())

# ------------------------------------------------------------ plots
cv_df.plot(kind="bar", figsize=(10, 4.5), colormap="tab10")
plt.title("Model comparison (repeated 5-fold CV, Health_condition excluded)")
plt.ylim(0, 1); plt.xticks(rotation=15); plt.legend(loc="lower right", ncol=5, fontsize=8)
plt.tight_layout(); plt.savefig("reports/model_comparison.png", dpi=130); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, (name, cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues"); ax.set_title(name, fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Lower", "Higher"]); ax.set_yticklabels(["Lower", "Higher"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("reports/confusion_matrices.png", dpi=130); plt.close()

# ------------------------------------------------------------ select + save
candidates = cv_df.drop(index=BASE)
best = candidates["roc_auc"].idxmax()
print(f"\nSelected model: {best} (highest CV ROC-AUC)")
final = make_pipe(models()[best], num_cols, cat_cols).fit(X, y)
joblib.dump(final, C.MODEL_PATH)

# Out-of-fold probabilities of the selected model: the web app converts a new
# student's probability into a percentile of this distribution (relative score).
oof = cross_val_predict(make_pipe(models()[best], num_cols, cat_cols), X, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                        method="predict_proba")[:, 1]
json.dump(sorted(round(float(p), 4) for p in oof), open(C.OOF_PATH, "w"))

rf = make_pipe(models()["Random Forest"], num_cols, cat_cols).fit(X, y)
imp = pd.Series(rf.named_steps["clf"].feature_importances_,
                index=rf.named_steps["prep"].get_feature_names_out())
imp.sort_values().tail(10).plot(kind="barh", figsize=(7, 4), color="#4c78a8")
plt.title("Top 10 feature importances (Random Forest)")
plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=130); plt.close()

rnd = lambda d: {k: {m: round(float(v), 3) for m, v in r.items()} for k, r in d.items()}
metrics = {
    "selected_model": best, "n_total": int(len(df)), "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)), "baseline": BASE,
    "cv": rnd(cv_mean), "cv_std": rnd(cv_std), "holdout": rnd(hold), "leakage": rnd(leak),
    "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
}
json.dump(metrics, open(C.METRICS_PATH, "w"), indent=2)
import os, shutil
os.makedirs("static/img", exist_ok=True)
for f in ["eda", "model_comparison", "confusion_matrices", "feature_importance"]:
    shutil.copy(f"reports/{f}.png", f"static/img/{f}.png")
print("Saved model, metrics, probabilities and plots.")
