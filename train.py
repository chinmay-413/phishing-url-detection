"""Train the phishing detector using BOTH datasets.

Dataset 1 (malicious_phish.csv)      : real URLs -> rich-feature model + URL-text model
Dataset 2 (phishing_url_dataset.csv) : numeric features only -> pooled with dataset 1 for a third model
The three models are combined by a small stacking model. Everything is saved to ./artifacts
"""
import json, os, re, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
from features import extract, legacy_features, text_norm, tld, logit, LEGACY_COLS

warnings.filterwarnings("ignore")

# ---------------- settings (edit if needed) ----------------
D1_FILE = "malicious_phish.csv"        # columns: url, type
D2_FILE = "phishing_url_dataset.csv"   # columns include LEGACY_COLS + target
N_BASE = 50000                         # URLs sampled from dataset 1 (x3 after augmentation)
FLIP_D2 = None                         # None = auto-detect; True/False to force the meaning of 'target'
SEED = 42
OUT = "artifacts"
# ------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(msg): print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def host_of(u):
    return re.sub(r"^[a-zA-Z]+://", "", str(u)).split("/")[0]


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred), "F1-score": f1_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, p), "PR-AUC": average_precision_score(y_true, p)}


def thin(a, b, k=200):
    idx = np.unique(np.linspace(0, len(a) - 1, min(k, len(a))).astype(int))
    return [round(float(x), 4) for x in np.asarray(a)[idx]], [round(float(x), 4) for x in np.asarray(b)[idx]]


# ---------- 1. dataset 1 ----------
log("Loading dataset 1 ...")
d1 = pd.read_csv(D1_FILE)
d1.columns = [c.lower().strip() for c in d1.columns]
d1 = d1[["url", "type"]].dropna()
d1["type"] = d1["type"].astype(str).str.lower().str.strip()
d1 = d1[d1["type"].isin(["phishing", "benign"])]
d1["label"] = (d1["type"] == "phishing").astype(int)
d1 = d1[["url", "label"]].drop_duplicates("url")
n_d1_full = len(d1)
d1 = d1.sample(min(len(d1), N_BASE), random_state=SEED).reset_index(drop=True)

# augmentation: also learn from the bare-domain and root-page versions (removes the "short URL = phishing" shortcut)
a1 = d1.copy(); a1["url"] = d1["url"].apply(host_of)
a2 = d1.copy(); a2["url"] = d1["url"].apply(lambda u: "https://" + host_of(u) + "/")
df = pd.concat([d1, a1, a2]).drop_duplicates("url").reset_index(drop=True)
y = df["label"]
log(f"Dataset 1: {len(d1)} URLs sampled -> {len(df)} after augmentation ({y.mean():.1%} phishing)")

# ---------- 2. features ----------
log("Extracting features (a few minutes) ...")
R = pd.DataFrame([extract(u) for u in df["url"]])
L = pd.DataFrame([legacy_features(u) for u in df["url"]])[LEGACY_COLS]
T = [text_norm(u) for u in df["url"]]
cols = list(R.columns)
groups = df["url"].apply(lambda u: tld(u if "://" in u else "http://" + u).top_domain_under_public_suffix or u)

# domain-separated split: train / validation / test never share a website
trv, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(R, y, groups))
a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(R.iloc[trv], y.iloc[trv], groups.iloc[trv]))
tr, va = trv[a], trv[b]
shared = len(set(groups.iloc[tr]) & set(groups.iloc[te]))
log(f"Split -> train {len(tr)} | val {len(va)} | test {len(te)} | domains shared train/test: {shared}")

# ---------- 3. dataset 2 ----------
log("Loading dataset 2 ...")
d2 = pd.read_csv(D2_FILE).dropna()
missing = [c for c in LEGACY_COLS + ["target"] if c not in d2.columns]
if missing:
    raise SystemExit(f"Dataset 2 is missing columns: {missing}\nColumns found: {list(d2.columns)}")
d2 = d2[LEGACY_COLS + ["target"]].drop_duplicates().reset_index(drop=True)

# the dataset does not say which value of 'target' means phishing -> check against dataset 1
probe = RandomForestClassifier(100, random_state=SEED, n_jobs=-1).fit(L.iloc[tr], y.iloc[tr])
auc_probe = roc_auc_score(d2["target"], probe.predict_proba(d2[LEGACY_COLS])[:, 1])
flip = (auc_probe < 0.5) if FLIP_D2 is None else FLIP_D2
log(f"Dataset 2 label check: AUC={auc_probe:.3f} -> target {'0' if flip else '1'} is treated as phishing"
    + ("   (WARNING: unclear, set FLIP_D2 manually)" if abs(auc_probe - 0.5) < 0.05 else ""))
d2["label"] = (1 - d2["target"]) if flip else d2["target"]
d2_tr, d2_te = train_test_split(d2, test_size=0.2, stratify=d2["label"], random_state=SEED)
print("\nShared feature means (should be similar in scale between the datasets):")
print(pd.DataFrame({"dataset1": L.mean(), "dataset2": d2[LEGACY_COLS].mean()}).round(2).to_string(), "\n")

# ---------- 4. models ----------
log("Training baseline models (Decision Tree, Logistic Regression, Random Forest) ...")
Rtr, ytr = R.iloc[tr], y.iloc[tr]
base = {
    "Decision Tree": DecisionTreeClassifier(max_depth=14, class_weight="balanced", random_state=SEED),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random Forest": RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
}
for m in base.values():
    m.fit(Rtr, ytr)

log("Training Gradient Boosting (model A: rich URL features) ...")
A = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, l2_regularization=1.0,
                                   class_weight="balanced", random_state=SEED).fit(Rtr, ytr)

log("Training character n-gram model (model B: URL text) ...")
B = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=300000,
                                  sublinear_tf=True, dtype=np.float32),
                  LogisticRegression(C=3, solver="liblinear", class_weight="balanced"))
B.fit([T[i] for i in tr], ytr)

log("Training pooled model (model C: dataset 1 + dataset 2 together) ...")
Xc = pd.concat([L.iloc[tr], d2_tr[LEGACY_COLS]])
yc = pd.concat([ytr, d2_tr["label"]])
C = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1).fit(Xc, yc)

# ---------- 5. ensemble ----------
def base_probs(idx):
    return np.c_[A.predict_proba(R.iloc[idx])[:, 1],
                 B.predict_proba([T[i] for i in idx])[:, 1],
                 C.predict_proba(L.iloc[idx])[:, 1]]

Pva, Pte = base_probs(va), base_probs(te)
S = LogisticRegression(class_weight="balanced").fit(logit(Pva), y.iloc[va])
ens_va, ens_te = S.predict_proba(logit(Pva))[:, 1], S.predict_proba(logit(Pte))[:, 1]
pr, rc, th = precision_recall_curve(y.iloc[va], ens_va)
f1s = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-9)
good = th[:len(f1s)][f1s >= f1s.max() - 0.002]          # plateau of near-best thresholds
thr = float(np.clip(np.median(good), 0.2, 0.8))           # keep the cut-off in a sensible range
log(f"Ensemble decision threshold (best F1 on validation): {thr:.3f}")

# ---------- 6. evaluation on unseen domains ----------
yte = y.iloc[te]
probs = {n: m.predict_proba(R.iloc[te])[:, 1] for n, m in base.items()}
probs.update({"Gradient Boosting": Pte[:, 0], "URL-text n-grams": Pte[:, 1],
              "Pooled RF (both datasets)": Pte[:, 2]})
rows = {n: metrics(yte, p) for n, p in probs.items()}
rows["ENSEMBLE (final)"] = metrics(yte, ens_te, thr)
table = pd.DataFrame(rows).T.round(4)
table.index.name = "Model"
table.to_csv(f"{OUT}/metrics.csv")
probs["ENSEMBLE (final)"] = ens_te

pC_d2 = C.predict_proba(d2_te[LEGACY_COLS])[:, 1]
d2_metrics = {k: round(float(v), 4) for k, v in metrics(d2_te["label"], pC_d2).items()}

curves = {"roc": {}, "pr": {}}
for n, p in probs.items():
    f_, t_, _ = roc_curve(yte, p); curves["roc"][n] = dict(zip(("x", "y"), thin(f_, t_)))
    p_, r_, _ = precision_recall_curve(yte, p); curves["pr"][n] = dict(zip(("x", "y"), thin(r_, p_)))
json.dump(curves, open(f"{OUT}/curves.json", "w"))

cm = confusion_matrix(yte, (ens_te >= thr).astype(int))

# which features matter (permutation importance on validation set, model A)
sub = np.random.RandomState(SEED).choice(va, min(4000, len(va)), replace=False)
pi = permutation_importance(A, R.iloc[sub], y.iloc[sub], scoring="average_precision", n_repeats=3, random_state=SEED)
pd.DataFrame({"feature": cols, "importance": pi.importances_mean}).sort_values("importance", ascending=False).to_csv(f"{OUT}/importance.csv", index=False)

# error analysis + analysis sample
tdf = pd.DataFrame({"url": df["url"].iloc[te].values, "label": yte.values, "risk": ens_te})
fp = tdf[(tdf.label == 0) & (tdf.risk >= thr)].sort_values("risk", ascending=False).head(25).assign(kind="False alarm")
fn = tdf[(tdf.label == 1) & (tdf.risk < thr)].sort_values("risk").head(25).assign(kind="Missed phishing")
pd.concat([fp, fn]).to_csv(f"{OUT}/errors.csv", index=False)
samp = R.iloc[te].copy(); samp["label"] = yte.values
samp.sample(min(6000, len(samp)), random_state=SEED).to_csv(f"{OUT}/analysis_sample.csv", index=False)

json.dump({"d1_rows_available": int(n_d1_full), "d1_rows_used": int(len(d1)), "rows_after_augmentation": int(len(df)),
           "phishing_share": round(float(y.mean()), 4), "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
           "shared_domains": int(shared), "d2_rows": int(len(d2)), "d2_target_flipped": bool(flip),
           "d2_label_check_auc": round(float(auc_probe), 3), "threshold": round(thr, 4),
           "confusion": cm.tolist(), "pooled_on_d2_holdout": d2_metrics},
          open(f"{OUT}/summary.json", "w"), indent=1)

joblib.dump({"A": A, "B": B, "C": C, "S": S, "thr": thr, "cols": cols, "lcols": LEGACY_COLS,
             "baseline": R.iloc[tr][ytr.values == 0].median().to_dict()},
            f"{OUT}/bundle.joblib", compress=3)

print("\n===== RESULTS ON UNSEEN DOMAINS (phishing = positive class) =====")
print(table.to_string())
print("\nPooled model on dataset 2 hold-out:", d2_metrics)
log("Done. Now run:  streamlit run app.py")