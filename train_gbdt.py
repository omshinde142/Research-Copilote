import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
import faiss

def compute_multiclass_brier(y_true, y_prob, n_classes):
    one_hot = np.zeros((len(y_true), n_classes))
    for i, y in enumerate(y_true):
        one_hot[i, y] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))

def main():
    print("=" * 70)
    print("      TRAINING LIGHTGBM & XGBOOST CLASSIFIERS ON FULL SPLITS")
    print("=" * 70)

    data_dir = Path("data")
    papers_path = data_dir / "papers.pkl"
    splits_path = data_dir / "splits.json"
    index_path = data_dir / "index.faiss"

    if not papers_path.exists() or not splits_path.exists():
        raise FileNotFoundError("Missing papers.pkl or splits.json.")

    df = pd.read_pickle(papers_path)
    splits = json.loads(splits_path.read_text())

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    classes = sorted(list(df.iloc[train_idx].primary_category.unique()))
    c2i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    print(f"Loaded {len(df)} papers across {n_classes} target classes.")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    y_train = np.array([c2i[c] for c in df.iloc[train_idx].primary_category])
    y_val = np.array([c2i[c] for c in df.iloc[val_idx].primary_category])
    y_test = np.array([c2i[c] for c in df.iloc[test_idx].primary_category])

    # Reconstruct dense vectors from index
    if index_path.exists():
        print(f"Reconstructing {len(df)} embeddings from {index_path}...")
        index = faiss.read_index(str(index_path))
        X_all = np.zeros((len(df), index.d), dtype="float32")
        for i in range(len(df)):
            X_all[i] = index.reconstruct(i)
    else:
        from sentence_transformers import SentenceTransformer
        print("Re-encoding embeddings with all-MiniLM-L6-v2...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        text = (df.title + ". " + df.abstract).tolist()
        X_all = model.encode(text, batch_size=64, normalize_embeddings=True)

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    results = {}

    # 1. LightGBM
    try:
        import lightgbm as lgb
        print("\nTraining LightGBM Classifier (100 estimators, lr=0.05)...")
        t0 = time.time()
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        clf_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        fit_time = round(time.time() - t0, 2)
        
        preds_val = clf_lgb.predict(X_val)
        preds_test = clf_lgb.predict(X_test)
        probas_test = clf_lgb.predict_proba(X_test)

        val_acc = round(float(accuracy_score(y_val, preds_val)), 4)
        test_acc = round(float(accuracy_score(y_test, preds_test)), 4)
        test_f1 = round(float(f1_score(y_test, preds_test, average="macro", zero_division=0)), 4)
        test_brier = round(compute_multiclass_brier(y_test, probas_test, n_classes), 4)

        results["LightGBM"] = {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_brier_score": test_brier,
            "fit_time_s": fit_time
        }
        print(f"  [LightGBM] Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Macro-F1: {test_f1:.4f} | Brier: {test_brier:.4f} ({fit_time}s)")
    except Exception as exc:
        print(f"  [LightGBM] Failed or not installed: {exc}")

    # 2. XGBoost
    try:
        import xgboost as xgb
        print("\nTraining XGBoost Classifier (100 estimators, lr=0.05)...")
        t0 = time.time()
        # Enable GPU if available
        device = "cuda" if xgb.rabit.get_rank() >= 0 else "cpu"
        clf_xgb = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=10
        )
        clf_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        fit_time = round(time.time() - t0, 2)

        preds_val = clf_xgb.predict(X_val)
        preds_test = clf_xgb.predict(X_test)
        probas_test = clf_xgb.predict_proba(X_test)

        val_acc = round(float(accuracy_score(y_val, preds_val)), 4)
        test_acc = round(float(accuracy_score(y_test, preds_test)), 4)
        test_f1 = round(float(f1_score(y_test, preds_test, average="macro", zero_division=0)), 4)
        test_brier = round(compute_multiclass_brier(y_test, probas_test, n_classes), 4)

        results["XGBoost"] = {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "test_brier_score": test_brier,
            "fit_time_s": fit_time
        }
        print(f"  [XGBoost]  Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | Macro-F1: {test_f1:.4f} | Brier: {test_brier:.4f} ({fit_time}s)")
    except Exception as exc:
        print(f"  [XGBoost] Failed or not installed: {exc}")

    out_file = data_dir / "gbdt_metrics.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved GBDT metrics to {out_file}")

if __name__ == "__main__":
    main()
