import re
import time
import unicodedata

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

DATASET = "clapAI/MultiLingualSentiment"
LANG = "en"


# 1. LOAD + FILTER
def load_split(split_name: str) -> tuple[list[str], np.ndarray]:
    ds = load_dataset(DATASET, split=split_name)
    n_before = len(ds)
    ds = ds.filter(
        lambda row: row["language"] == LANG,
        num_proc=4,
        desc=f"  filter lang={LANG} [{split_name}]",
    )
    print(f"  [{split_name}] {n_before:,} → {len(ds):,} rows")
    texts = list(tqdm(ds["text"], desc=f"  load [{split_name}]", dynamic_ncols=True))
    labels = np.array(ds["label"])
    return texts, labels


# 2. PREPROCESS
def preprocess(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_batch(texts: list[str], split_name: str) -> list[str]:
    return [
        preprocess(t)
        for t in tqdm(texts, desc=f"  preprocess [{split_name}]", dynamic_ncols=True)
    ]


# 3. VECTORIZE


def vectorize(VecClass, X_train, X_test, name):
    vec = VecClass()
    print(f"  [{name}] fit_transform train …")
    t0 = time.perf_counter()
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    print(
        f"    vocab={len(vec.vocabulary_):,}  shape={X_tr.shape}  ({time.perf_counter() - t0:.1f}s)"
    )
    return vec, X_tr, X_te


# 4. TRAIN + EVALUATE


def run(name, model, X_tr, X_te, y_train, y_test, vec):
    print(f"\n  ── {name} ──")
    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"    acc={acc:.4f}  f1={f1:.4f}  time={train_time:.1f}s")
    return {
        "name": name,
        "model": model,
        "vec": vec,
        "accuracy": acc,
        "f1": f1,
        "train_time": train_time,
        "y_pred": y_pred,
    }


# 5. COMPARE
def print_comparison(results, y_test):
    W = 72
    print("\n" + "═" * W)
    print("  COMPARISON SUMMARY")
    print("═" * W)
    print(f"  {'Model':<32} {'Accuracy':>10} {'F1 (w)':>10} {'Train (s)':>10}")
    print("  " + "─" * (W - 2))

    best_acc = max(r["accuracy"] for r in results)
    best_f1 = max(r["f1"] for r in results)
    best_spd = min(r["train_time"] for r in results)

    for r in results:
        print(
            f"  {r['name']:<32}"
            f"  {r['accuracy']:>8.4f}{'★' if r['accuracy'] == best_acc else ' '}"
            f"  {r['f1']:>8.4f}{'★' if r['f1'] == best_f1 else ' '}"
            f"  {r['train_time']:>8.1f}{'★' if r['train_time'] == best_spd else ' '}"
        )
    print("═" * W)
    print("  ★ = best in column\n")

    for r in results:
        print(f"\n── Classification Report: {r['name']} ──")
        print(classification_report(y_test, r["y_pred"], digits=4))


# 6. DEMO
DEMO_TEXTS = [
    "I absolutely love this product, it changed my life!",
    "Terrible experience, would not recommend to anyone.",
    "It is okay, nothing special but gets the job done.",
    "This movie was a complete waste of time.",
    "Best purchase I have ever made, highly recommend!",
]


def demo_predict(results):
    print("\n── Demo Predictions ──")
    col = 20
    header = f"  {'Text':<46}" + "".join(f"  {r['name'][:col]:<{col}}" for r in results)
    print(header)
    print("  " + "─" * (46 + (col + 2) * len(results)))
    for text in DEMO_TEXTS:
        clean = preprocess(text)
        row = f"  {text[:44]:<46}"
        for r in results:
            X = r["vec"].transform([clean])
            row += f"  {str(r['model'].predict(X)[0]):<{col}}"
        print(row)


def main():
    # 1. Load
    print("\n[1] Load …")
    X_train_raw, y_train = load_split("train")
    X_test_raw, y_test = load_split("test")

    # 2. Preprocess
    print("\n[2] Preprocess …")
    X_train = preprocess_batch(X_train_raw, "train")
    X_test = preprocess_batch(X_test_raw, "test")

    # 3. Vectorize
    print("\n[3] Vectorize …")
    bow_vec, X_tr_bow, X_te_bow = vectorize(CountVectorizer, X_train, X_test, "BoW")
    tfidf_vec, X_tr_tfidf, X_te_tfidf = vectorize(
        TfidfVectorizer, X_train, X_test, "TF-IDF"
    )

    # 3b. Lưu vocab
    import json

    for name, vec in [("bow", bow_vec), ("tfidf", tfidf_vec)]:
        path = f"{name}_vocab.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vec.vocabulary_, f, ensure_ascii=False, indent=2)
        print(f"  saved {path}  ({len(vec.vocabulary_):,} tokens)")

    # 4. Train + Evaluate
    print("\n[4] Train …")
    results = [
        run(
            "LR + BoW",
            LogisticRegression(max_iter=1000),
            X_tr_bow,
            X_te_bow,
            y_train,
            y_test,
            bow_vec,
        ),
        run(
            "LR + TF-IDF",
            LogisticRegression(max_iter=1000),
            X_tr_tfidf,
            X_te_tfidf,
            y_train,
            y_test,
            tfidf_vec,
        ),
        run("NB + BoW", MultinomialNB(), X_tr_bow, X_te_bow, y_train, y_test, bow_vec),
        run(
            "NB + TF-IDF",
            MultinomialNB(),
            X_tr_tfidf,
            X_te_tfidf,
            y_train,
            y_test,
            tfidf_vec,
        ),
    ]

    # 5. Compare
    print_comparison(results, y_test)

    # 6. Demo
    demo_predict(results)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
