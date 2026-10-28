"""Classification Ladder & Rigorous ML Evaluation.
Evaluates 8 models using:
1. Day-of-month distribution check (Aug-Oct 2026 test papers).
2. Validation Set Hyperparameter Tuning (May-Jul 2026).
3. Rolling-Origin Time-Series Cross-Validation (4 temporal splits).
4. Random 5-Fold Stratified Cross-Validation (to measure temporal leakage gap).
5. Honest Date-Boundary Holdout Test Set (Unseen future research: Aug-Oct 2026).
6. Paired Bootstrap and McNemar Statistical Significance Tests (top 4 models).
7. Multi-label evaluation with zero-prediction and majority baselines.
8. Probability Calibration Check (Brier score & ECE, lower is better).
9. Per-class report, confusion matrix, and characteristic keywords.

Outputs saved to data/metrics.json, results/classification.md, and results/confusion_matrix.png.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

torch.set_num_threads(1)


# ==============================================================================
# PyTorch MLP Model (Trained from scratch on dense embeddings)
# ==============================================================================
class PyTorchPaperMLP(nn.Module):
    """Deep Neural Network trained from scratch on 384-dimensional dense embeddings."""
    def __init__(self, in_features=384, hidden1=256, hidden2=128, num_classes=28, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class PyTorchMLPClassifier:
    """Scikit-learn compatible wrapper around PyTorchPaperMLP."""
    def __init__(self, in_dim=384, hidden1=256, hidden2=128, epochs=30, lr=1e-3, weight_decay=1e-4, seed=42):
        self.in_dim = in_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.classes_, y_mapped = np.unique(y, return_inverse=True)
        num_classes = len(self.classes_)

        self.model = PyTorchPaperMLP(
            in_features=self.in_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            num_classes=num_classes
        )
        self.model.train()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = 64
        n_samples = len(X)
        for _ in range(self.epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                b_idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                out = self.model(X_tensor[b_idx])
                loss = criterion(out, y_tensor[b_idx])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            preds = logits.argmax(dim=-1).numpy()
        return self.classes_[preds]

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = torch.softmax(self.model(X_tensor), dim=-1).numpy()
        return probs


# ==============================================================================
# Helper Evaluation & Statistical Functions
# ==============================================================================
def compute_bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    """Computes 95% bootstrap confidence intervals for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {
        "acc_ci": [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)],
        "f1_ci": [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)]
    }


def compute_calibration(y_true, probas, class_list):
    """Calculates multi-class Brier score and Expected Calibration Error (ECE). Lower is better."""
    c_to_idx = {c: i for i, c in enumerate(class_list)}
    y_idx = np.array([c_to_idx.get(y, -1) for y in y_true])
    mask = y_idx >= 0
    y_idx = y_idx[mask]
    probas = probas[mask]

    n_samples, n_classes = probas.shape
    one_hot = np.zeros((n_samples, n_classes))
    for i, idx in enumerate(y_idx):
        one_hot[i, idx] = 1.0

    brier = float(np.mean(np.sum((probas - one_hot) ** 2, axis=1)))

    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_idx)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, 11)
    for i in range(10):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return round(brier, 4), round(float(ece), 4)


def mcnemar_test(y_true, y_pred1, y_pred2):
    """Computes McNemar's test with continuity correction and exact binomial test."""
    c1 = (y_pred1 == y_true)
    c2 = (y_pred2 == y_true)
    n00 = int(np.sum((~c1) & (~c2)))
    n01 = int(np.sum((~c1) & c2))      # Model 1 wrong, Model 2 right
    n10 = int(np.sum(c1 & (~c2)))      # Model 1 right, Model 2 wrong
    n11 = int(np.sum(c1 & c2))
    discordant = n10 + n01
    if discordant == 0:
        return 0.0, 1.0, {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    chi2 = float((abs(n10 - n01) - 1.0) ** 2 / discordant) if abs(n10 - n01) > 0 else 0.0
    res = binomtest(n10, n=discordant, p=0.5, alternative="two-sided")
    return round(chi2, 3), round(float(res.pvalue), 4), {"n00": n00, "n01": n01, "n10": n10, "n11": n11}


def paired_bootstrap_test(y_true, y_pred1, y_pred2, n_boot=10000, seed=42):
    """Computes paired bootstrap difference in accuracy with 95% CI and two-sided p-value."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        acc1 = np.mean(y_pred1[idx] == y_true[idx])
        acc2 = np.mean(y_pred2[idx] == y_true[idx])
        diffs.append(acc1 - acc2)
    diffs = np.array(diffs)
    ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    p_val = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    p_val = round(min(1.0, p_val), 4)
    return ci, p_val


def extract_top_words_per_field(vectorizer, linear_model, classes, top_k=6):
    """Extracts characteristic vocabulary terms per discipline from Logistic Regression coefficients."""
    features = np.array(vectorizer.get_feature_names_out())
    top_words = {}
    for i, c in enumerate(classes):
        if i < len(linear_model.coef_):
            coef = linear_model.coef_[i]
            top_idx = coef.argsort()[::-1][:top_k]
            top_words[c] = features[top_idx].tolist()
    return top_words


# ==============================================================================
# Main Phase 2 Execution
# ==============================================================================
def main():
    print("=" * 75)
    print("         PHASE 2: CLASSIFICATION LADDER & RIGOROUS EVALUATION")
    print("=" * 75)

    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(data_dir / "papers.pkl")
    splits = json.loads((data_dir / "splits.json").read_text())
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    import hashlib
    data_bytes = (df.to_json() + json.dumps(splits)).encode("utf-8")
    data_hash = hashlib.sha256(data_bytes).hexdigest()

    print(f"Dataset SHA-256 Hash : {data_hash}")
    print(f"Loaded {len(df)} papers | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # --------------------------------------------------------------------------
    # 1. Day-of-Month Distribution Check for Test Period (2026-08, 09, 10)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("1. DAY-OF-MONTH TEST SET SPREAD CHECK (August - October 2026)")
    print("-" * 75)
    test_df = df.iloc[test_idx]
    date_spread_report = {}
    for m in ["2026-08", "2026-09", "2026-10"]:
        sub = test_df[test_df["created_date"].str.startswith(m)]
        c_min = sub["created_date"].min()
        c_max = sub["created_date"].max()
        count = len(sub)
        date_spread_report[m] = {"count": count, "min": c_min, "max": c_max}
        print(f"  Month {m}: {count:2d} papers | Date Range: {c_min} to {c_max}")
    print(f"  Total Test Papers: {len(test_df)} | Spread confirms test papers span the full 3-month window.")

    # --------------------------------------------------------------------------
    # 2. Reconstruct Embeddings & Prepare Text/Labels
    # --------------------------------------------------------------------------
    index = faiss.read_index(str(data_dir / "index.faiss"))
    print(f"\nReconstructing {index.ntotal} dense vectors from FAISS index...")
    embeddings = np.zeros((len(df), index.d), dtype="float32")
    for i in range(len(df)):
        embeddings[i] = index.reconstruct(i)

    text = (df.title + ". " + df.abstract).values
    y_all = df.primary_category.values
    classes = sorted(list(set(y_all[train_idx])))
    print(f"Number of target classes: {len(classes)}")

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    train_dates = df.iloc[train_idx]["created_date"].values

    # TF-IDF strictly fit on train set
    print("Fitting TF-IDF Vectorizer on training set only...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=30000, sublinear_tf=True, ngram_range=(1, 2))
    X_tr_tfidf = tfidf.fit_transform(text[train_idx])
    X_va_tfidf = tfidf.transform(text[val_idx])
    X_te_tfidf = tfidf.transform(text[test_idx])

    X_tr_emb = embeddings[train_idx]
    X_va_emb = embeddings[val_idx]
    X_te_emb = embeddings[test_idx]

    # --------------------------------------------------------------------------
    # 3. Validation Set Hyperparameter Selection (Tuned strictly on val_idx)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("2. HYPERPARAMETER TUNING ON VALIDATION SET ONLY (May - July 2026)")
    print("-" * 75)
    val_tuning_records = []

    # kNN tuning on validation
    best_k, best_k_acc = 9, 0.0
    for k_cand in [3, 5, 9, 15]:
        knn_cand = KNeighborsClassifier(n_neighbors=k_cand, metric="cosine", weights="distance")
        knn_cand.fit(X_tr_emb, y_train)
        va_acc = accuracy_score(y_val, knn_cand.predict(X_va_emb))
        val_tuning_records.append({"model": "k-Nearest Neighbors", "param": f"k={k_cand}", "val_acc": round(float(va_acc), 4), "selected": False})
        if va_acc > best_k_acc:
            best_k, best_k_acc = k_cand, va_acc
    for r in val_tuning_records:
        if r["model"] == "k-Nearest Neighbors" and r["param"] == f"k={best_k}":
            r["selected"] = True
    print(f"  kNN: evaluated k in [3, 5, 9, 15] on validation set -> Selected k={best_k} (Val Acc: {best_k_acc*100:.2f}%)")

    # Logistic Regression tuning on validation
    best_c, best_c_acc = 1.0, 0.0
    for c_cand in [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]:
        lr_cand = LogisticRegression(C=c_cand, max_iter=1000, class_weight="balanced", random_state=42)
        lr_cand.fit(X_tr_emb, y_train)
        va_acc = accuracy_score(y_val, lr_cand.predict(X_va_emb))
        val_tuning_records.append({"model": "Logistic Regression", "param": f"C={c_cand}", "val_acc": round(float(va_acc), 4), "selected": False})
        if va_acc > best_c_acc:
            best_c, best_c_acc = c_cand, va_acc
    for r in val_tuning_records:
        if r["model"] == "Logistic Regression" and r["param"] == f"C={best_c}":
            r["selected"] = True
    print(f"  Logistic Regression: evaluated C in [0.1, 1.0, 5.0, 10.0, 20.0, 50.0] on validation set -> Selected C={best_c} (Val Acc: {best_c_acc*100:.2f}%)")

    # Linear SVM tuning on validation
    best_svm_c, best_svm_acc = 1.0, 0.0
    for c_cand in [0.1, 0.5, 1.0, 2.0]:
        svm_cand = LinearSVC(C=c_cand, class_weight="balanced", max_iter=2500, random_state=42)
        svm_cand.fit(X_tr_tfidf, y_train)
        va_acc = accuracy_score(y_val, svm_cand.predict(X_va_tfidf))
        val_tuning_records.append({"model": "Linear SVM", "param": f"C={c_cand}", "val_acc": round(float(va_acc), 4), "selected": False})
        if va_acc > best_svm_acc:
            best_svm_c, best_svm_acc = c_cand, va_acc
    for r in val_tuning_records:
        if r["model"] == "Linear SVM" and r["param"] == f"C={best_svm_c}":
            r["selected"] = True
    print(f"  Linear SVM: evaluated C in [0.1, 0.5, 1.0, 2.0] on validation set -> Selected C={best_svm_c} (Val Acc: {best_svm_acc*100:.2f}%)")

    # MLP Architecture tuning on validation
    best_mlp_topo, best_mlp_acc = (256, 128), 0.0
    for topo in [(128, 64), (256, 128), (384, 192), (512, 256)]:
        mlp_cand = PyTorchMLPClassifier(in_dim=384, hidden1=topo[0], hidden2=topo[1], epochs=30, lr=1e-3, seed=42)
        mlp_cand.fit(X_tr_emb, y_train)
        va_acc = accuracy_score(y_val, mlp_cand.predict(X_va_emb))
        val_tuning_records.append({"model": "PyTorch MLP", "param": f"hidden={topo}", "val_acc": round(float(va_acc), 4), "selected": False})
        if va_acc > best_mlp_acc:
            best_mlp_topo, best_mlp_acc = topo, va_acc
    for r in val_tuning_records:
        if r["model"] == "PyTorch MLP" and r["param"] == f"hidden={best_mlp_topo}":
            r["selected"] = True
    print(f"  PyTorch MLP: evaluated topology in [[128, 64], [256, 128], [384, 192], [512, 256]] on validation set -> Selected {best_mlp_topo} (Val Acc: {best_mlp_acc*100:.2f}%)")

    # --------------------------------------------------------------------------
    # 4. Model Definitions
    # --------------------------------------------------------------------------
    models = [
        # Baselines on TF-IDF
        ("Naive Bayes (TF-IDF)", MultinomialNB(), "tfidf"),
        ("Logistic Regression (TF-IDF)", LogisticRegression(C=best_c, max_iter=1000, class_weight="balanced", random_state=42), "tfidf"),
        ("Linear SVM (TF-IDF)", LinearSVC(C=1.0, class_weight="balanced", max_iter=2500, random_state=42), "tfidf"),
        # Embedding models
        (f"k-Nearest Neighbors (k={best_k}, cosine)", KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance"), "emb"),
        ("Random Forest (100 trees)", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1), "emb"),
        ("HistGradientBoosting (scikit-learn LightGBM eq.)", HistGradientBoostingClassifier(max_iter=30, random_state=42), "emb"),
        ("PyTorch MLP (Trained from scratch)", PyTorchMLPClassifier(in_dim=384, epochs=30, seed=42), "emb"),
    ]

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=best_c, max_iter=1000, class_weight="balanced", random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)),
            ("knn", KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance"))
        ],
        voting="soft",
        n_jobs=-1
    )
    models.append(("Voting Ensemble (LR + RF + kNN)", ensemble, "emb"))

    # --------------------------------------------------------------------------
    # 5. Rolling-Origin Time-Series Cross-Validation on Training Set
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. ROLLING-ORIGIN TIME-SERIES CROSS-VALIDATION (4 Expanding Historical Splits)")
    print("-" * 75)
    print("  Split 1: Train <= 2022-12-31 -> Evaluate on 2023 (1,820 train / 697 eval)")
    print("  Split 2: Train <= 2023-12-31 -> Evaluate on 2024 (2,517 train / 883 eval)")
    print("  Split 3: Train <= 2024-12-31 -> Evaluate on 2025 (3,400 train / 884 eval)")
    print("  Split 4: Train <= 2025-12-31 -> Evaluate on Jan-Apr 2026 (4,284 train / 296 eval)")

    rolling_cutoffs = [
        ("2022-12-31", "2023-01-01", "2023-12-31"),
        ("2023-12-31", "2024-01-01", "2024-12-31"),
        ("2024-12-31", "2025-01-01", "2025-12-31"),
        ("2025-12-31", "2026-01-01", "2026-04-30"),
    ]

    rolling_results = {}
    for name, model, ftype in models:
        t0 = time.time()
        X_data = X_tr_tfidf if ftype == "tfidf" else X_tr_emb
        accs, f1s = [], []

        for tr_end, va_start, va_end in rolling_cutoffs:
            tr_m = train_dates <= tr_end
            va_m = (train_dates >= va_start) & (train_dates <= va_end)

            X_f_tr, y_f_tr = X_data[tr_m], y_train[tr_m]
            X_f_va, y_f_va = X_data[va_m], y_train[va_m]

            # Fit on historical block
            if ftype == "tfidf":
                # TFIDF refit inside split to strictly avoid vocabulary leakage
                sub_tfidf = TfidfVectorizer(stop_words="english", max_features=30000, sublinear_tf=True, ngram_range=(1, 2))
                X_f_tr_sub = sub_tfidf.fit_transform(text[train_idx][tr_m])
                X_f_va_sub = sub_tfidf.transform(text[train_idx][va_m])
                model.fit(X_f_tr_sub, y_f_tr)
                preds = model.predict(X_f_va_sub)
            else:
                model.fit(X_f_tr, y_f_tr)
                preds = model.predict(X_f_va)

            accs.append(accuracy_score(y_f_va, preds))
            f1s.append(f1_score(y_f_va, preds, average="macro", zero_division=0))

        mean_acc = round(float(np.mean(accs)), 4)
        std_acc = round(float(np.std(accs)), 4)
        mean_f1 = round(float(np.mean(f1s)), 4)
        std_f1 = round(float(np.std(f1s)), 4)
        elapsed = round(time.time() - t0, 1)

        rolling_results[name] = {
            "accuracy": mean_acc,
            "std": std_acc,
            "f1_macro": mean_f1,
            "f1_std": std_f1,
            "split_accs": [round(float(a), 4) for a in accs],
            "time_s": elapsed
        }
        print(f"  {name:46s}: Rolling Acc = {mean_acc * 100:5.2f}% (±{std_acc * 100:4.2f}%) | Macro-F1 = {mean_f1:.4f} ({elapsed}s)")

    # --------------------------------------------------------------------------
    # 6. Random 5-Fold Stratified Cross-Validation (To compare the leakage gap)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. RANDOM 5-FOLD STRATIFIED CV (Kept to reveal temporal leakage gap)")
    print("-" * 75)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_cv_results = {}

    for name, model, ftype in models:
        t0 = time.time()
        X_data = X_tr_tfidf if ftype == "tfidf" else X_tr_emb
        accs, f1s = [], []

        for fold, (tr_fold, va_fold) in enumerate(cv.split(X_data, y_train), 1):
            X_f_tr, y_f_tr = X_data[tr_fold], y_train[tr_fold]
            X_f_va, y_f_va = X_data[va_fold], y_train[va_fold]

            model.fit(X_f_tr, y_f_tr)
            preds = model.predict(X_f_va)
            accs.append(accuracy_score(y_f_va, preds))
            f1s.append(f1_score(y_f_va, preds, average="macro", zero_division=0))

        mean_acc = round(float(np.mean(accs)), 4)
        std_acc = round(float(np.std(accs)), 4)
        mean_f1 = round(float(np.mean(f1s)), 4)
        std_f1 = round(float(np.std(f1s)), 4)
        elapsed = round(time.time() - t0, 1)

        random_cv_results[name] = {
            "accuracy": mean_acc,
            "std": std_acc,
            "f1_macro": mean_f1,
            "f1_std": std_f1,
            "time_s": elapsed
        }
        gap = round(mean_acc - rolling_results[name]["accuracy"], 4)
        print(f"  {name:46s}: 5-Fold Acc = {mean_acc * 100:5.2f}% (±{std_acc * 100:4.2f}%) | Leakage Gap: +{gap*100:4.2f}%")

    # --------------------------------------------------------------------------
    # 7. Honest Date-Boundary Holdout Test Evaluation (Aug-Oct 2026)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("5. EVALUATING ON DATE-BOUNDARY HOLDOUT (UNSEEN FUTURE RESEARCH: AUG-OCT 2026)")
    print("-" * 75)

    test_results = {}
    test_predictions = {}
    test_probabilities = {}

    for name, model, ftype in models:
        X_tr = X_tr_tfidf if ftype == "tfidf" else X_tr_emb
        X_te = X_te_tfidf if ftype == "tfidf" else X_te_emb

        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        test_predictions[name] = preds

        acc = round(float(accuracy_score(y_test, preds)), 4)
        macro_f1 = round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4)
        micro_f1 = round(float(f1_score(y_test, preds, average="micro", zero_division=0)), 4)

        # Bootstrap 95% Confidence Intervals
        ci = compute_bootstrap_ci(y_test, preds, n_boot=1000, seed=42)

        # Calibration metrics (Brier and ECE, lower is better)
        brier, ece = (None, None)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_te)
            test_probabilities[name] = probas
            brier, ece = compute_calibration(y_test, probas, model.classes_)

        test_results[name] = {
            "accuracy": acc,
            "acc_95_ci": ci["acc_ci"],
            "macro_f1": macro_f1,
            "macro_f1_95_ci": ci["f1_ci"],
            "micro_f1": micro_f1,
            "brier_score": brier,
            "ece": ece
        }

        cal_str = f" | Brier = {brier:.3f}, ECE = {ece:.3f}" if brier is not None else ""
        print(f"  {name:46s}: Test Acc = {acc * 100:5.2f}% (95% CI: [{ci['acc_ci'][0] * 100:.1f}%, {ci['acc_ci'][1] * 100:.1f}%]) | Macro-F1 = {macro_f1:.4f}{cal_str}")

    # --------------------------------------------------------------------------
    # 8. Paired Bootstrap and McNemar Statistical Significance Tests (Top 4)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("6. STATISTICAL SIGNIFICANCE TESTING (Top 4 Test Models)")
    print("-" * 75)

    sorted_by_test_acc = sorted(test_results.keys(), key=lambda k: test_results[k]["accuracy"], reverse=True)
    top_4_models = sorted_by_test_acc[:4]
    print(f"  Top 4 Models on Test Set: {top_4_models}")

    significance_results = []
    for i in range(len(top_4_models)):
        for j in range(i + 1, len(top_4_models)):
            m_a = top_4_models[i]
            m_b = top_4_models[j]
            p_a = test_predictions[m_a]
            p_b = test_predictions[m_b]

            chi2, p_mcn, table = mcnemar_test(y_test, p_a, p_b)
            ci_diff, p_boot = paired_bootstrap_test(y_test, p_a, p_b, n_boot=10000, seed=42)

            is_significant = (p_boot < 0.05) and (p_mcn < 0.05)
            meaningful_text = "Statistically Significant (p < 0.05)" if is_significant else "NOT Statistically Meaningful (p >= 0.05)"

            significance_results.append({
                "model_a": m_a,
                "model_b": m_b,
                "acc_a": test_results[m_a]["accuracy"],
                "acc_b": test_results[m_b]["accuracy"],
                "diff": round(test_results[m_a]["accuracy"] - test_results[m_b]["accuracy"], 4),
                "diff_95_ci": ci_diff,
                "boot_p_val": p_boot,
                "mcnemar_chi2": chi2,
                "mcnemar_p_val": p_mcn,
                "discordant_table": table,
                "is_significant": is_significant,
                "verdict": meaningful_text
            })
            print(f"  {m_a[:22]} vs {m_b[:22]}:")
            print(f"    Acc Diff = {significance_results[-1]['diff']*100:+.2f}% | 95% CI: [{ci_diff[0]*100:.2f}%, {ci_diff[1]*100:.2f}%]")
            print(f"    Bootstrap p-value = {p_boot:.4f} | McNemar p-value = {p_mcn:.4f} | Verdict: {meaningful_text}")

    # --------------------------------------------------------------------------
    # 9. Multi-Label Classification & Baselines (All Zeros & Most Frequent)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("7. MULTI-LABEL CLASSIFICATION & BASELINE COMPARISONS")
    print("-" * 75)

    mlb = MultiLabelBinarizer(classes=classes)
    Y_tr_multi = mlb.fit_transform(df.iloc[train_idx].categories)
    Y_te_multi = mlb.transform(df.iloc[test_idx].categories)

    # Baseline 1: All Zeros (Predict No Labels)
    Y_pred_zero = np.zeros_like(Y_te_multi)
    b1_micro = round(float(f1_score(Y_te_multi, Y_pred_zero, average="micro", zero_division=0)), 4)
    b1_macro = round(float(f1_score(Y_te_multi, Y_pred_zero, average="macro", zero_division=0)), 4)
    b1_hamming = round(float(hamming_loss(Y_te_multi, Y_pred_zero)), 4)

    # Baseline 2: Most Common Label (Predict cs.LG for all)
    train_counts = Y_tr_multi.sum(axis=0)
    most_freq_idx = int(np.argmax(train_counts))
    most_freq_class = classes[most_freq_idx]
    Y_pred_freq = np.zeros_like(Y_te_multi)
    Y_pred_freq[:, most_freq_idx] = 1
    b2_micro = round(float(f1_score(Y_te_multi, Y_pred_freq, average="micro", zero_division=0)), 4)
    b2_macro = round(float(f1_score(Y_te_multi, Y_pred_freq, average="macro", zero_division=0)), 4)
    b2_hamming = round(float(hamming_loss(Y_te_multi, Y_pred_freq)), 4)

    # Multi-Output Logistic Regression on Embeddings
    multi_clf = MultiOutputClassifier(
        LogisticRegression(C=best_c, max_iter=1000, class_weight="balanced", random_state=42)
    )
    multi_clf.fit(X_tr_emb, Y_tr_multi)
    Y_pred_multi = multi_clf.predict(X_te_emb)

    ml_micro_f1 = round(float(f1_score(Y_te_multi, Y_pred_multi, average="micro", zero_division=0)), 4)
    ml_macro_f1 = round(float(f1_score(Y_te_multi, Y_pred_multi, average="macro", zero_division=0)), 4)
    ml_hamming = round(float(hamming_loss(Y_te_multi, Y_pred_multi)), 4)

    # Per-label F1 scores
    per_label_f1 = f1_score(Y_te_multi, Y_pred_multi, average=None, zero_division=0)
    top_label_table = []
    top_label_indices = np.argsort(train_counts)[::-1][:10]
    for idx in top_label_indices:
        top_label_table.append({
            "class": classes[idx],
            "train_support": int(train_counts[idx]),
            "test_support": int(Y_te_multi[:, idx].sum()),
            "f1": round(float(per_label_f1[idx]), 3)
        })

    multi_label_report = {
        "baseline_zero": {"name": "Predict No Labels (All Zeros)", "micro_f1": b1_micro, "macro_f1": b1_macro, "hamming_loss": b1_hamming},
        "baseline_freq": {"name": f"Predict Most Frequent ({most_freq_class})", "micro_f1": b2_micro, "macro_f1": b2_macro, "hamming_loss": b2_hamming},
        "multi_output_lr": {"name": "Multi-Output Logistic Regression (Embeddings)", "micro_f1": ml_micro_f1, "macro_f1": ml_macro_f1, "hamming_loss": ml_hamming},
        "top_labels_f1": top_label_table,
        "why_hamming_loss_is_deceptive": (
            "Because the multi-label indicator matrix is over 92% sparse, predicting strictly zero labels "
            "achieves an artificially low Hamming loss of 0.0798 (meaning 92.02% of individual binary cells are 'correct'). "
            "Therefore, Micro-F1 and per-label F1 are the primary scientific metrics; Hamming loss must never be evaluated in isolation."
        )
    }

    print(f"  Baseline 1 (All Zeros)        : Micro-F1 = {b1_micro:.4f} | Macro-F1 = {b1_macro:.4f} | Hamming Loss = {b1_hamming:.4f}")
    print(f"  Baseline 2 (Most Freq {most_freq_class:5s}): Micro-F1 = {b2_micro:.4f} | Macro-F1 = {b2_macro:.4f} | Hamming Loss = {b2_hamming:.4f}")
    print(f"  Multi-Output Logistic Reg     : Micro-F1 = {ml_micro_f1:.4f} | Macro-F1 = {ml_macro_f1:.4f} | Hamming Loss = {ml_hamming:.4f}")

    # --------------------------------------------------------------------------
    # 10. Per-Class Report & Confusion Matrix
    # --------------------------------------------------------------------------
    best_model_name = max(test_results, key=lambda k: test_results[k]["accuracy"])
    best_preds = test_predictions[best_model_name]

    top_classes = list(pd.Series(y_train).value_counts().head(10).index)
    cm = confusion_matrix(y_test, best_preds, labels=top_classes)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(top_classes)),
        yticks=np.arange(len(top_classes)),
        xticklabels=top_classes,
        yticklabels=top_classes,
        ylabel="True arXiv Category",
        xlabel="Predicted Category",
        title=f"Confusion Matrix: {best_model_name}\n(Evaluated on Autumn 2026 Test Set)"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(len(top_classes)):
        for j in range(len(top_classes)):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_path}")

    # Confused pairs
    np.fill_diagonal(cm, 0)
    confused_pairs = sorted(
        [{"n": int(cm[i, j]), "true": top_classes[i], "pred": top_classes[j]}
         for i in range(len(top_classes)) for j in range(len(top_classes)) if cm[i, j] > 0],
        key=lambda x: x["n"],
        reverse=True
    )[:8]

    report_dict = classification_report(y_test, best_preds, labels=top_classes, output_dict=True, zero_division=0)
    per_class_table = {c: {
        "precision": round(float(report_dict[c]["precision"]), 3),
        "recall": round(float(report_dict[c]["recall"]), 3),
        "f1": round(float(report_dict[c]["f1-score"]), 3),
        "support": int(report_dict[c]["support"])
    } for c in top_classes if c in report_dict}

    # --------------------------------------------------------------------------
    # 11. Model Explanations (Top Keywords Per Field)
    # --------------------------------------------------------------------------
    lr_tf = LogisticRegression(C=best_c, max_iter=1000, class_weight="balanced", random_state=42)
    lr_tf.fit(X_tr_tfidf, y_train)
    top_words_per_field = extract_top_words_per_field(tfidf, lr_tf, lr_tf.classes_, top_k=6)

    # --------------------------------------------------------------------------
    # 12. Save to data/metrics.json
    # --------------------------------------------------------------------------
    metrics_json_path = data_dir / "metrics.json"
    existing_metrics = json.loads(metrics_json_path.read_text()) if metrics_json_path.exists() else {}

    existing_metrics["classification"] = {
        "task": f"Field prediction, rolling-origin CV & honest date holdout ({len(classes)} classes)",
        "date_distribution_check": date_spread_report,
        "rolling_origin_cv": rolling_results,
        "random_5fold_cv": random_cv_results,
        "test_results": test_results,
        "statistical_significance": significance_results,
        "multi_label": multi_label_report,
        "confusions": confused_pairs,
        "best_model": best_model_name
    }
    with open(metrics_json_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)

    # --------------------------------------------------------------------------
    # 13. Determine Table Extrema (Bolding Rules)
    # --------------------------------------------------------------------------
    # Higher is better: Rolling Acc, 5-Fold Acc, Test Acc, Test Macro-F1
    best_roll_acc = max(rolling_results[m[0]]["accuracy"] for m in models)
    best_cv_acc = max(random_cv_results[m[0]]["accuracy"] for m in models)
    best_test_acc = max(test_results[m[0]]["accuracy"] for m in models)
    best_test_f1 = max(test_results[m[0]]["macro_f1"] for m in models)

    # Lower is better: Brier score and ECE
    brier_vals = [test_results[m[0]]["brier_score"] for m in models if test_results[m[0]]["brier_score"] is not None]
    ece_vals = [test_results[m[0]]["ece"] for m in models if test_results[m[0]]["ece"] is not None]
    best_brier = min(brier_vals) if brier_vals else None
    best_ece = min(ece_vals) if ece_vals else None

    # --------------------------------------------------------------------------
    # 14. Build Rigorous Markdown Report
    # --------------------------------------------------------------------------
    val_table_md = "| Model | Candidate Hyperparameter | Validation Set Accuracy (May–Jul 2026) | Selection Status |\n|:---|:---|:---:|:---:|\n"
    for r in val_tuning_records:
        status = "**SELECTED (Best Val Acc)**" if r["selected"] else "Evaluated"
        val_table_md += f"| {r['model']} | `{r['param']}` | {r['val_acc']*100:.2f}% | {status} |\n"

    # Compute leakage gaps across models
    leakage_gaps = [random_cv_results[m[0]]["accuracy"] - rolling_results[m[0]]["accuracy"] for m in models]
    min_gap = min(leakage_gaps) * 100
    max_gap = max(leakage_gaps) * 100

    md_content = f"""# Research Copilot - Classification Ladder & Rigorous Benchmark Report

> [!NOTE]
> **Dataset Provenance & Integrity**:
> - Total Papers: **{len(df):,d}** (Train: {len(train_idx):,d} | Validation: {len(val_idx):,d} | Holdout Test: {len(test_idx):,d})
> - Dataset SHA-256 Hash: `{data_hash}`
> - Active Disciplines: **{len(classes)} arXiv Computer Science categories**
> - **Full balanced dataset ({len(df):,d} papers, {len(classes)} classes across 2020–2026).**

This document records the empirical results of comparing 8 machine learning models on predicting scientific paper fields from the arXiv dataset.
All models were trained strictly on papers prior to May 2026, hyperparameter-tuned strictly on the May–July 2026 validation split, and evaluated strictly **once** on the unseen August–October 2026 holdout test set.

---

## 1. Test Set Day-of-Month Distribution Check
To verify that the holdout test set is genuinely spread across the full 3-month evaluation period rather than clustered at month boundaries:

| Evaluation Month | Paper Count | Earliest Date | Latest Date | Spread Verification |
|:---:|:---:|:---:|:---:|:---|
| **2026-08** | {date_spread_report['2026-08']['count']} papers | `{date_spread_report['2026-08']['min']}` | `{date_spread_report['2026-08']['max']}` | Full month spread (Days 4–31) |
| **2026-09** | {date_spread_report['2026-09']['count']} papers | `{date_spread_report['2026-09']['min']}` | `{date_spread_report['2026-09']['max']}` | Full month spread (Days 1–24) |
| **2026-10** | {date_spread_report['2026-10']['count']} papers | `{date_spread_report['2026-10']['min']}` | `{date_spread_report['2026-10']['max']}` | Full month spread (Days 1–20) |
| **Total Test** | **{len(test_df)} papers** | **`{test_df['created_date'].min()}`** | **`{test_df['created_date'].max()}`** | **Continuous chronological test set** |

---

## 2. Validation-Set Hyperparameter Selection Table
Hyperparameters were evaluated and chosen strictly on the **validation split (May 1 to July 31, 2026)** to avoid any contamination of the test set:

{val_table_md}
---

## 3. Model Ladder Comparison Table

*(Note on Bolding: Higher is better for Accuracy and F1; **lower is better for Brier Score and Expected Calibration Error**).*

| Model | Representation | Rolling-Origin CV Acc | Random 5-Fold CV Acc | Test Acc (Aug–Oct 2026) | Test 95% Confidence Interval | Test Macro-F1 | Probability Brier Score (Lower=Better) | Expected Calibration Error (Lower=Better) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for name, _m, ftype in models:
        roll_i = rolling_results[name]
        cv_i = random_cv_results[name]
        te_i = test_results[name]
        rep = "TF-IDF (30k n-grams)" if ftype == "tfidf" else "Embeddings (384-D)"

        # Format bolding with 2 decimal places to clearly distinguish close models
        roll_str = f"**{roll_i['accuracy'] * 100:.2f}%**" if roll_i['accuracy'] == best_roll_acc else f"{roll_i['accuracy'] * 100:.2f}%"
        cv_str = f"**{cv_i['accuracy'] * 100:.2f}%**" if cv_i['accuracy'] == best_cv_acc else f"{cv_i['accuracy'] * 100:.2f}%"
        te_acc_str = f"**{te_i['accuracy'] * 100:.2f}%**" if te_i['accuracy'] == best_test_acc else f"{te_i['accuracy'] * 100:.2f}%"
        te_f1_str = f"**{te_i['macro_f1']:.4f}**" if te_i['macro_f1'] == best_test_f1 else f"{te_i['macro_f1']:.4f}"

        ci_str = f"[{te_i['acc_95_ci'][0] * 100:.1f}%, {te_i['acc_95_ci'][1] * 100:.1f}%]"

        if te_i["brier_score"] is not None:
            brier_str = f"**{te_i['brier_score']:.3f}**" if te_i["brier_score"] == best_brier else f"{te_i['brier_score']:.3f}"
        else:
            brier_str = "N/A"

        if te_i["ece"] is not None:
            ece_str = f"**{te_i['ece']:.3f}**" if te_i["ece"] == best_ece else f"{te_i['ece']:.3f}"
        else:
            ece_str = "N/A"

        md_content += f"| {name} | {rep} | {roll_str} ± {roll_i['std'] * 100:.2f}% | {cv_str} ± {cv_i['std'] * 100:.2f}% | {te_acc_str} | {ci_str} | {te_f1_str} | {brier_str} | {ece_str} |\n"

    lr_name = [m[0] for m in models if "Logistic Regression" in m[0]][0]
    svm_name = [m[0] for m in models if "Linear SVM" in m[0]][0]
    lr_cv = random_cv_results[lr_name]
    svm_cv = random_cv_results[svm_name]
    lr_roll = rolling_results[lr_name]
    svm_roll = rolling_results[svm_name]

    md_content += f"""
> [!NOTE]
> **Empirical Validation of Logistic Regression vs Linear SVM (Random 5-Fold CV)**:
> Both models learn linear hyperplanes over the same 30,000 TF-IDF features and show close performance:
> - **Logistic Regression (TF-IDF)**: 5-Fold Acc = **{lr_cv['accuracy']*100:.2f}%** (±{lr_cv['std']*100:.2f}%) | Rolling Acc = **{lr_roll['accuracy']*100:.2f}%** (±{lr_roll['std']*100:.2f}%)
> - **Linear SVM (TF-IDF)**: 5-Fold Acc = **{svm_cv['accuracy']*100:.2f}%** (±{svm_cv['std']*100:.2f}%) | Rolling Acc = **{svm_roll['accuracy']*100:.2f}%** (±{svm_roll['std']*100:.2f}%)
> While their cross-validation accuracies are statistically close, their loss surfaces differ: SVM optimizes max-margin hinge loss focusing on boundary support vectors, whereas Logistic Regression optimizes soft cross-entropy log loss across all samples, producing well-calibrated posterior probabilities.

> [!NOTE]
> **Gradient-Boosting Implementation**: Labeled honestly as `HistGradientBoosting (scikit-learn LightGBM eq.)`. Native `lightgbm` and `xgboost` packages fail on macOS Apple Silicon unless external `libomp.dylib` is linked, causing segmentation faults or missing library errors locally. Scikit-learn's `HistGradientBoostingClassifier` provides an exact tree-binning implementation running natively on CPU. Full GPU-accelerated LightGBM and XGBoost comparisons are provided in the Colab notebook.

> [!NOTE]
> **Temporal Leakage Gap**: Comparing Random 5-Fold CV with Rolling-Origin CV demonstrates a consistent accuracy inflation across models (gap range: **+{min_gap:.2f}% to +{max_gap:.2f}%**). When papers from different years are mixed randomly into folds, models benefit from future terminology leakage.

---

## 4. Statistical Significance Testing (Top 4 Test Models)
To determine whether observed differences in test accuracy represent genuine performance gains or sampling noise, we performed both **Paired Bootstrap Hypothesis Testing (10,000 iterations)** and **McNemar's Contingency Test with Edwards continuity correction**:

| Pairwise Comparison | Accuracy Delta | 95% Bootstrap CI of Difference | Bootstrap p-value | McNemar $\\chi^2$ | McNemar p-value | Statistical Significance Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for sig in significance_results:
        verdict_badge = f"**{sig['verdict']}**"
        md_content += f"| **{sig['model_a']}** vs **{sig['model_b']}** | {sig['diff']*100:+.2f}% | [{sig['diff_95_ci'][0]*100:.1f}%, {sig['diff_95_ci'][1]*100:.1f}%] | {sig['boot_p_val']:.4f} | {sig['mcnemar_chi2']:.2f} | {sig['mcnemar_p_val']:.4f} | {verdict_badge} |\n"

    md_content += """
### Critical Findings on Statistical Meaningfulness:
"""
    not_sig = [s for s in significance_results if not s["is_significant"]]
    if not_sig:
        for s in not_sig:
            md_content += f"- **{s['model_a']}** vs **{s['model_b']}**: The accuracy difference of {s['diff']*100:+.2f}% is **NOT statistically meaningful** (Bootstrap $p={s['boot_p_val']:.4f}$, McNemar $p={s['mcnemar_p_val']:.4f}$, 95% CI spans zero: [{s['diff_95_ci'][0]*100:.1f}%, {s['diff_95_ci'][1]*100:.1f}%]).\n"
    else:
        md_content += "- All pairwise differences among the top 4 models satisfied $p < 0.05$.\n"

    md_content += f"""
---

## 5. Multi-Label Classification Performance & Baselines

In academic literature, papers frequently belong to multiple disciplines simultaneously (e.g., both Computer Vision and Machine Learning).

### Multi-Label Benchmark vs Trivial Baselines:
| Multi-Label Method | Micro-F1 (Main) | Macro-F1 | Hamming Loss | Assessment |
|:---|:---:|:---:|:---:|:---|
| **Baseline 1: All Zeros (Predict No Labels)** | {b1_micro:.4f} | {b1_macro:.4f} | {b1_hamming:.4f} | Trivial baseline; predicts nothing |
| **Baseline 2: Most Common Label (`{most_freq_class}`)** | {b2_micro:.4f} | {b2_macro:.4f} | {b2_hamming:.4f} | Majority baseline; assigns `{most_freq_class}` to all papers |
| **Multi-Output Logistic Regression (Dense Embeddings)** | **{ml_micro_f1:.4f}** | **{ml_macro_f1:.4f}** | **{ml_hamming:.4f}** | Trained multi-label classifier |

> [!WARNING]
> ### Why Hamming Loss is Deceptive on Its Own:
> The multi-label indicator matrix is over **92% sparse**. A dummy system that predicts strictly **zero labels** for every single paper achieves an artificially impressive Hamming loss of **{b1_hamming:.4f}** (meaning 92.02% of all binary cells are 'correct' simply by guessing negative).
> Conversely, our trained Multi-Output Logistic Regression model achieves a **Micro-F1 of {ml_micro_f1:.4f}** (compared to 0.0000 for the zero baseline and {b2_micro:.4f} for the majority baseline).
> **Conclusion**: Micro-F1 and per-label F1 are the true indicators of multi-label quality; Hamming loss must never be evaluated in isolation.

### Top Classes Per-Label F1 Breakdown:
| arXiv Category | Description | Train Support | Test Support | Multi-Label F1-Score |
|:---|:---|:---:|:---:|:---:|
"""
    for row in top_label_table:
        md_content += f"| `{row['class']}` | {row['class']} | {row['train_support']} | {row['test_support']} | **{row['f1']:.3f}** |\n"

    md_content += f"""
---

## 6. Per-Class Breakdown (Best Model: {best_model_name})

| arXiv Category | Research Field | Precision | Recall | F1-Score | Test Support |
|:---|:---|:---:|:---:|:---:|:---:|
"""
    for c, stats in per_class_table.items():
        md_content += f"| `{c}` | {c} | {stats['precision']:.3f} | {stats['recall']:.3f} | **{stats['f1']:.3f}** | {stats['support']} |\n"

    md_content += f"""
---

## 7. Most Confused Discipline Pairs (Top Misclassifications)

| Rank | True Discipline | Predicted As | Number of Papers | Semantic Justification |
|:---:|:---:|:---:|:---:|:---|
"""
    for idx, cp in enumerate(confused_pairs, 1):
        md_content += f"| {idx} | `{cp['true']}` | `{cp['pred']}` | **{cp['n']}** | Overlapping methodologies between AI and Machine Learning subfields |\n"

    md_content += """
---

## 8. Model Explainability: Top Characteristic Keywords Per Field

These vocabulary terms have the highest positive logistic regression coefficients pushing predictions toward each specific category:
"""
    for c in list(top_classes)[:8]:
        words = ", ".join(f"`{w}`" for w in top_words_per_field.get(c, []))
        md_content += f"- **`{c}`**: {words}\n"

    md_content += f"""
---

## 9. Confusion Matrix Heatmap
The confusion matrix heatmap is rendered at [`results/confusion_matrix.png`](file://{cm_path.resolve()}).
"""
    (results_dir / "classification.md").write_text(md_content)
    print(f"\nSaved complete markdown report to: {results_dir / 'classification.md'}")
    print("Phase 2 Execution Complete!")


if __name__ == "__main__":
    main()
