import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "chiaki_products.csv"
SEARCH_COLUMNS = ["name", "category", "description"]

df = pd.read_csv(DATA_PATH)
df = df.fillna("")

df["search_text"] = df[SEARCH_COLUMNS].astype(str).agg(" ".join, axis=1)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
)

tfidf_matrix = vectorizer.fit_transform(df["search_text"])

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
joblib.dump(df, "products.pkl")

print("Build xong.")
print("Total products:", len(df))
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Matrix shape:", tfidf_matrix.shape)
