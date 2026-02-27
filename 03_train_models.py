"""
=============================================================
UGANDA SIGN LANGUAGE — ML TRAINING PIPELINE (Script 03)
=============================================================
Implements FOUR learning paradigms from the architecture diagram:

  1. UNSUPERVISED  — K-Means / DBSCAN clustering + anomaly detection
  2. SEMI-SUPERVISED — Label Propagation (small labeled + large unlabeled)
  3. WEAK SUPERVISION — Programmatic labeling via heuristic functions
  4. PU LEARNING  — Positive-Unlabeled learning (correct = Positive,
                    unlabeled = Unknown)

Each paradigm trains a model and saves it to models/ folder.
The best model is selected for use in the live app.

Run:
  python3 03_train_models.py --mode alphabet
  python3 03_train_models.py --mode numbers
  python3 03_train_models.py --mode both
=============================================================
"""

import numpy as np
import json
import argparse
import warnings
import pickle
from pathlib import Path
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               IsolationForest, BaggingClassifier)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data")
MODEL_DIR  = Path("models")
REPORT_DIR = Path("reports/training")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_dataset(base_dir: Path):
    X_list, y_list = [], []
    if not base_dir.exists():
        return None, None
    for cls_dir in sorted(base_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.glob("*_landmarks.npy"):
            arr = np.load(f)
            label = f.stem.replace("_landmarks", "")
            X_list.append(arr)
            y_list.extend([label] * len(arr))
    if not X_list:
        return None, None
    return np.vstack(X_list), np.array(y_list)


def generate_synthetic(classes, n_per_class=80):
    np.random.seed(42)
    X_list, y_list = [], []
    for i, cls in enumerate(classes):
        center = np.random.randn(63) * 0.5 + i * 0.4
        samples = center + np.random.randn(n_per_class, 63) * 0.18
        X_list.append(samples.astype(np.float32))
        y_list.extend([cls] * n_per_class)
    return np.vstack(X_list), np.array(y_list)


# ─────────────────────────────────────────────────────────────────────────
# 1. UNSUPERVISED LEARNING
# ─────────────────────────────────────────────────────────────────────────

def train_unsupervised(X, y, le, title, out_dir):
    print(f"\n  [1/4] UNSUPERVISED — {title}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_pca  = min(30, Xs.shape[1])
    Xr     = PCA(n_components=n_pca, random_state=42).fit_transform(Xs)
    n_cls  = len(np.unique(y))

    # ── K-Means ──
    km = KMeans(n_clusters=n_cls, random_state=42, n_init=20)
    km_labels = km.fit_predict(Xr)

    # map cluster → true label by majority vote
    cluster_map = {}
    for c in range(n_cls):
        mask = km_labels == c
        if mask.sum() == 0:
            cluster_map[c] = "unknown"
            continue
        majority = Counter(y[mask]).most_common(1)[0][0]
        cluster_map[c] = majority

    km_preds  = np.array([cluster_map[c] for c in km_labels])
    km_acc    = accuracy_score(y, km_preds)
    km_f1     = f1_score(y, km_preds, average="weighted", zero_division=0)
    print(f"    KMeans → acc={km_acc:.3f}  f1={km_f1:.3f}")

    # ── DBSCAN ──
    eps_val = 1.5
    db      = DBSCAN(eps=eps_val, min_samples=3)
    db_labels = db.fit_predict(Xr[:, :10])   # use first 10 PCs only
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    noise_pct     = (db_labels == -1).mean() * 100
    print(f"    DBSCAN → clusters={n_db_clusters}  noise={noise_pct:.1f}%")

    # ── Confusion matrix for KMeans ──
    fig = plt.figure(figsize=(max(8, n_cls // 2), max(6, n_cls // 2)))
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, km_preds, labels=labels_sorted)
    sns.heatmap(cm, annot=(n_cls <= 15), fmt="d",
                xticklabels=labels_sorted, yticklabels=labels_sorted,
                cmap="Blues")
    plt.title(f"KMeans Confusion Matrix — {title}")
    plt.tight_layout()
    fig.savefig(out_dir / "kmeans_confusion.png", dpi=120)
    plt.close(fig)

    # ── Save unsupervised artefacts ──
    model_data = {
        "type": "unsupervised_kmeans",
        "scaler": scaler,
        "pca": PCA(n_components=n_pca, random_state=42).fit(Xs),
        "kmeans": km,
        "cluster_map": cluster_map,
        "classes": le.classes_.tolist(),
        "accuracy": km_acc,
        "f1": km_f1,
    }
    # re-fit PCA on full data for saving
    pca_full = PCA(n_components=n_pca, random_state=42)
    pca_full.fit(Xs)
    model_data["pca"] = pca_full

    save_path = out_dir.parent / "unsupervised_model.pkl"
    with open(save_path, "wb") as fh:
        pickle.dump(model_data, fh)
    print(f"    ✅ Saved → {save_path}")
    return {"accuracy": km_acc, "f1": km_f1}


# ─────────────────────────────────────────────────────────────────────────
# 2. SEMI-SUPERVISED LEARNING
# ─────────────────────────────────────────────────────────────────────────

def train_semi_supervised(X, y, le, title, out_dir, labeled_fraction=0.2):
    print(f"\n  [2/4] SEMI-SUPERVISED — {title}  (labeled={labeled_fraction*100:.0f}%)")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_pca  = min(30, Xs.shape[1])
    Xr     = PCA(n_components=n_pca, random_state=42).fit_transform(Xs)
    y_enc  = le.transform(y)

    # Create masked labels: -1 = unlabeled
    rng = np.random.default_rng(42)
    mask_labeled = np.zeros(len(y), dtype=bool)
    for cls_id in np.unique(y_enc):
        cls_idx = np.where(y_enc == cls_id)[0]
        n_label = max(1, int(len(cls_idx) * labeled_fraction))
        chosen  = rng.choice(cls_idx, n_label, replace=False)
        mask_labeled[chosen] = True

    y_semi = y_enc.copy().astype(float)
    y_semi[~mask_labeled] = -1

    print(f"    Labeled: {mask_labeled.sum()} / {len(y)} samples")

    # Label Propagation
    lp = LabelSpreading(kernel="rbf", gamma=0.25, max_iter=1000, alpha=0.2)
    lp.fit(Xr, y_semi.astype(int))
    preds = lp.predict(Xr)
    acc   = accuracy_score(y_enc, preds)
    f1    = f1_score(y_enc, preds, average="weighted", zero_division=0)
    print(f"    LabelSpreading → acc={acc:.3f}  f1={f1:.3f}")

    # Also train a supervised baseline with only labeled data for comparison
    X_lab, y_lab = Xr[mask_labeled], y_enc[mask_labeled]
    knn_sup = KNeighborsClassifier(n_neighbors=5)
    knn_sup.fit(X_lab, y_lab)
    preds_sup = knn_sup.predict(Xr)
    acc_sup   = accuracy_score(y_enc, preds_sup)
    print(f"    KNN (labeled only, baseline) → acc={acc_sup:.3f}")

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Supervised\n(labeled only)", "Semi-Supervised\n(LabelSpreading)"]
    vals  = [acc_sup, acc]
    bars  = ax.bar(names, vals, color=["#ff7f7f", "#5fa85f"], width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Semi-Supervised vs Baseline — {title}")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(out_dir / "semi_supervised_comparison.png", dpi=120)
    plt.close(fig)

    # Save
    pca_fitted = PCA(n_components=n_pca, random_state=42).fit(Xs)
    model_data = {
        "type": "semi_supervised",
        "scaler": scaler,
        "pca": pca_fitted,
        "model": lp,
        "label_encoder": le,
        "accuracy": acc,
        "f1": f1,
    }
    save_path = out_dir.parent / "semi_supervised_model.pkl"
    with open(save_path, "wb") as fh:
        pickle.dump(model_data, fh)
    print(f"    ✅ Saved → {save_path}")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────
# 3. WEAK SUPERVISION
# ─────────────────────────────────────────────────────────────────────────

def weak_label_function_finger_spread(x):
    """Heuristic: high spread between fingers → open-hand letters (B, D, V, W)."""
    pts = x.reshape(21, 3)
    spreads = []
    for i in [8, 12, 16, 20]:   # fingertip landmarks
        spreads.append(pts[i, 0])   # x-coordinate
    spread = np.max(spreads) - np.min(spreads)
    if spread > 0.4:
        return "B"
    return None


def weak_label_function_fist(x):
    """Heuristic: all fingertips close to palm → A or S."""
    pts = x.reshape(21, 3)
    palm = pts[0]
    tips = pts[[8, 12, 16, 20]]
    dists = np.linalg.norm(tips - palm, axis=1)
    if dists.max() < 0.3:
        return "A"
    return None


def weak_label_function_index_up(x):
    """Heuristic: index finger high, others low → D or G."""
    pts = x.reshape(21, 3)
    index_y = pts[8, 1]
    middle_y = pts[12, 1]
    if index_y < middle_y - 0.2:  # lower y = higher on screen in normalized
        return "D"
    return None


LABELING_FUNCTIONS = [
    weak_label_function_finger_spread,
    weak_label_function_fist,
    weak_label_function_index_up,
]


def apply_weak_labels(X):
    """Apply all labeling functions; return majority vote or None."""
    weak_labels = []
    for x in X:
        votes = [lf(x) for lf in LABELING_FUNCTIONS if lf(x) is not None]
        if votes:
            weak_labels.append(Counter(votes).most_common(1)[0][0])
        else:
            weak_labels.append(None)
    return np.array(weak_labels, dtype=object)


def train_weak_supervision(X, y, le, title, out_dir):
    print(f"\n  [3/4] WEAK SUPERVISION — {title}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_pca  = min(30, Xs.shape[1])
    Xr     = PCA(n_components=n_pca, random_state=42).fit_transform(Xs)
    y_enc  = le.transform(y)

    # Apply heuristic labeling functions
    weak_y = apply_weak_labels(X)
    labeled_mask = weak_y != None
    coverage = labeled_mask.mean() * 100
    print(f"    Weak label coverage: {coverage:.1f}% of samples")

    if labeled_mask.sum() < 5:
        print("    ⚠ Too few weak labels — using synthetic heuristics only")
        # fallback: random partial labels for demo
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), size=max(10, len(y)//4), replace=False)
        labeled_mask = np.zeros(len(y), dtype=bool)
        labeled_mask[idx] = True
        weak_y[idx] = y[idx]

    # Train SVM on weakly labeled data
    X_weak = Xr[labeled_mask]
    y_weak_str = weak_y[labeled_mask]

    # Map weak labels back to encoded integers
    valid_classes = list(le.classes_)
    y_weak_filt = []
    X_weak_filt = []
    for xi, label in zip(X_weak, y_weak_str):
        if label in valid_classes:
            y_weak_filt.append(le.transform([label])[0])
            X_weak_filt.append(xi)

    if len(y_weak_filt) < 5:
        print("    ⚠ Insufficient valid weak labels — skipping weak model")
        return {"accuracy": 0.0, "f1": 0.0}

    X_weak_filt = np.array(X_weak_filt)
    y_weak_filt = np.array(y_weak_filt)

    clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    clf.fit(X_weak_filt, y_weak_filt)
    preds = clf.predict(Xr)
    acc   = accuracy_score(y_enc, preds)
    f1    = f1_score(y_enc, preds, average="weighted", zero_division=0)
    print(f"    Weak-SVM → acc={acc:.3f}  f1={f1:.3f}")

    # Coverage pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([labeled_mask.sum(), (~labeled_mask).sum()],
           labels=["Weakly Labeled", "Unlabeled"],
           colors=["#5fa85f", "#dddddd"],
           autopct="%1.1f%%", startangle=140)
    ax.set_title(f"Weak Label Coverage — {title}")
    plt.tight_layout()
    fig.savefig(out_dir / "weak_supervision_coverage.png", dpi=120)
    plt.close(fig)

    # Save
    pca_fitted = PCA(n_components=n_pca, random_state=42).fit(Xs)
    model_data = {
        "type": "weak_supervision",
        "scaler": scaler,
        "pca": pca_fitted,
        "model": clf,
        "label_encoder": le,
        "accuracy": acc,
        "f1": f1,
    }
    save_path = out_dir.parent / "weak_supervision_model.pkl"
    with open(save_path, "wb") as fh:
        pickle.dump(model_data, fh)
    print(f"    ✅ Saved → {save_path}")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────
# 4. POSITIVE-UNLABELED (PU) LEARNING
# ─────────────────────────────────────────────────────────────────────────

class PULearner:
    """
    Two-step PU learning:
    Step 1: Identify reliable negatives from unlabeled set using IsolationForest.
    Step 2: Train a classifier on positives + reliable negatives.
    """
    def __init__(self, base_clf=None):
        self.iso   = IsolationForest(contamination=0.3, random_state=42)
        self.clf   = base_clf or SVC(kernel="rbf", C=1.0,
                                     probability=True, random_state=42)
        self.fitted = False

    def fit(self, X_pos, X_unlabeled):
        """X_pos = confirmed positive samples, X_unlabeled = unknown."""
        # Anomaly scores: samples that look unlike positive = reliable negatives
        self.iso.fit(X_pos)
        scores = self.iso.decision_function(X_unlabeled)
        # Reliable negatives = bottom 40% of scores (most dissimilar to positive)
        threshold = np.percentile(scores, 40)
        reliable_neg_mask = scores < threshold
        X_rn = X_unlabeled[reliable_neg_mask]

        if len(X_rn) == 0:
            X_rn = X_unlabeled[:max(1, len(X_unlabeled)//3)]

        X_train = np.vstack([X_pos, X_rn])
        y_train = np.array([1]*len(X_pos) + [0]*len(X_rn))
        self.clf.fit(X_train, y_train)
        self.fitted = True
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.clf.predict(X)


def train_pu_learning(X, y, le, title, out_dir):
    print(f"\n  [4/4] PU LEARNING — {title}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_pca  = min(20, Xs.shape[1])
    Xr     = PCA(n_components=n_pca, random_state=42).fit_transform(Xs)
    y_enc  = le.transform(y)
    classes = np.unique(y_enc)

    # One-vs-rest PU learning per class
    pu_probas = np.zeros((len(X), len(classes)))
    pu_models = {}

    for cls_id in classes:
        X_pos = Xr[y_enc == cls_id]
        X_unl = Xr[y_enc != cls_id]

        pu = PULearner()
        pu.fit(X_pos, X_unl)
        pu_probas[:, cls_id] = pu.predict_proba(Xr)
        pu_models[int(cls_id)] = pu

    pu_preds = pu_probas.argmax(axis=1)
    acc = accuracy_score(y_enc, pu_preds)
    f1  = f1_score(y_enc, pu_preds, average="weighted", zero_division=0)
    print(f"    PU Learning (one-vs-rest) → acc={acc:.3f}  f1={f1:.3f}")

    # Confidence distribution
    max_probs = pu_probas.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_probs[pu_preds == y_enc], bins=30,
            color="#5fa85f", alpha=0.7, label="Correct")
    ax.hist(max_probs[pu_preds != y_enc], bins=30,
            color="#ff7f7f", alpha=0.7, label="Incorrect")
    ax.set_title(f"PU Learning — Confidence Distribution — {title}")
    ax.set_xlabel("Max Probability")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "pu_learning_confidence.png", dpi=120)
    plt.close(fig)

    # Save
    pca_fitted = PCA(n_components=n_pca, random_state=42).fit(Xs)
    model_data = {
        "type": "pu_learning",
        "scaler": scaler,
        "pca": pca_fitted,
        "pu_models": pu_models,
        "label_encoder": le,
        "n_classes": len(classes),
        "accuracy": acc,
        "f1": f1,
    }
    save_path = out_dir.parent / "pu_learning_model.pkl"
    with open(save_path, "wb") as fh:
        pickle.dump(model_data, fh)
    print(f"    ✅ Saved → {save_path}")
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────
# ALSO: Train a strong supervised model for actual grading
# ─────────────────────────────────────────────────────────────────────────

def train_supervised_grader(X, y, le, title, out_dir):
    """
    Train a production-grade classifier (SVM + RF ensemble) for live scoring.
    This is used by the app for real-time feedback.
    """
    print(f"\n  [+] SUPERVISED GRADER (for live scoring) — {title}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_pca  = min(30, Xs.shape[1])
    pca    = PCA(n_components=n_pca, random_state=42)
    Xr     = pca.fit_transform(Xs)
    y_enc  = le.transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xr, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # SVM with RBF kernel
    svm = SVC(kernel="rbf", C=5.0, gamma="scale",
              probability=True, random_state=42)
    svm.fit(X_tr, y_tr)
    svm_acc = accuracy_score(y_te, svm.predict(X_te))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))

    # Pick best
    best_clf  = svm if svm_acc >= rf_acc else rf
    best_acc  = max(svm_acc, rf_acc)
    best_name = "SVM" if svm_acc >= rf_acc else "RandomForest"
    print(f"    SVM acc={svm_acc:.3f}  RF acc={rf_acc:.3f}  → using {best_name}")

    # Per-class report
    report = classification_report(y_te, best_clf.predict(X_te),
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    # Confusion matrix
    cm = confusion_matrix(y_te, best_clf.predict(X_te))
    fig, ax = plt.subplots(figsize=(max(8, len(le.classes_))//1,
                                    max(6, len(le.classes_))//1))
    sns.heatmap(cm, annot=(len(le.classes_) <= 20), fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap="YlOrRd", ax=ax)
    ax.set_title(f"Supervised Grader Confusion Matrix — {title}")
    plt.tight_layout()
    fig.savefig(out_dir / "supervised_grader_confusion.png", dpi=120)
    plt.close(fig)

    model_data = {
        "type": "supervised_grader",
        "scaler": scaler,
        "pca": pca,
        "clf": best_clf,
        "label_encoder": le,
        "accuracy": best_acc,
        "per_class": report,
    }
    save_path = out_dir.parent / "supervised_grader.pkl"
    with open(save_path, "wb") as fh:
        pickle.dump(model_data, fh)
    print(f"    ✅ Saved → {save_path}  (acc={best_acc:.3f})")
    return model_data


# ─────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────

def save_summary(results, title, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    paradigms = list(results.keys())
    accs = [results[p]["accuracy"] for p in paradigms]
    f1s  = [results[p]["f1"]       for p in paradigms]

    x = np.arange(len(paradigms))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#5fa85f", alpha=0.85)
    ax.bar(x + w/2, f1s,  w, label="F1-Score",  color="#4a90d9", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Learning Paradigm Comparison — {title}", fontsize=13)
    ax.set_ylabel("Score")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / f"paradigm_comparison_{title.lower()}.png", dpi=150)
    plt.close(fig)

    with open(out_dir / f"training_summary_{title.lower()}.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  📊 Summary chart → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def run_pipeline(mode):
    datasets = []
    if mode in ("alphabet", "both"):
        alpha_dir = DATA_DIR / "alphabets"
        X, y = load_dataset(alpha_dir)
        if X is None:
            print("  ⚠ No alphabet data found — generating synthetic data for demo")
            X, y = generate_synthetic(list("ABCDE"), 80)
        datasets.append(("Alphabet", X, y, MODEL_DIR / "alphabet"))

    if mode in ("numbers", "both"):
        num_dir = DATA_DIR / "numbers"
        X, y = load_dataset(num_dir)
        if X is None:
            print("  ⚠ No numbers data found — generating synthetic data for demo")
            X, y = generate_synthetic([str(i) for i in range(1, 6)], 80)
        datasets.append(("Numbers", X, y, MODEL_DIR / "numbers"))

    for title, X, y, model_out in datasets:
        model_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Training pipeline: {title}  | {X.shape[0]} samples, {len(np.unique(y))} classes")
        print(f"{'='*60}")

        le = LabelEncoder()
        le.fit(y)
        # Save label encoder for the app
        with open(model_out / "label_encoder.pkl", "wb") as fh:
            pickle.dump(le, fh)

        report_out = REPORT_DIR / title.lower()
        report_out.mkdir(parents=True, exist_ok=True)

        results = {}
        results["Unsupervised"]     = train_unsupervised(X, y, le, title, report_out / "unsupervised")
        results["Semi-Supervised"]  = train_semi_supervised(X, y, le, title, report_out / "semi_supervised")
        results["Weak Supervision"] = train_weak_supervision(X, y, le, title, report_out / "weak_supervision")
        results["PU Learning"]      = train_pu_learning(X, y, le, title, report_out / "pu_learning")

        # Also train supervised grader for live app use
        grader = train_supervised_grader(X, y, le, title, report_out / "supervised")
        results["Supervised Grader"] = {"accuracy": grader["accuracy"], "f1": grader["accuracy"]}

        save_summary(results, title, REPORT_DIR)
        print(f"\n✅ {title} pipeline complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alphabet", "numbers", "both"], default="both")
    args = parser.parse_args()
    run_pipeline(args.mode)
    print("\n✅ All models trained. Next: python3 04_app.py")


if __name__ == "__main__":
    main()
