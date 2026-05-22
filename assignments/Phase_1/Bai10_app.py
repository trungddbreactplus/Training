import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

RESULT_COLUMNS = ["name", "category", "description"]


@st.cache_resource
def load_index():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    df = joblib.load("products.pkl")
    return vectorizer, tfidf_matrix, df


def search(query, vectorizer, tfidf_matrix, df, top_k=10):
    query_vec = vectorizer.transform([query.strip()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]

    mask = top_scores > 0
    top_indices = top_indices[mask]
    top_scores = top_scores[mask]

    if len(top_indices) == 0:
        return pd.DataFrame()

    show_cols = [c for c in RESULT_COLUMNS if c in df.columns]
    result = df.iloc[top_indices][show_cols].copy()
    result.insert(0, "score", np.round(top_scores, 4))
    result = result.reset_index(drop=True)

    return result


# -- UI --

st.set_page_config(page_title="Product Search", layout="wide")
st.title("Product Search")

vectorizer, tfidf_matrix, df = load_index()
st.caption(f"{len(df):,} sản phẩm · vocab {len(vectorizer.vocabulary_):,} terms")

query = st.text_input(
    "Tìm kiếm", placeholder="Nhập tên, danh mục hoặc mô tả sản phẩm..."
)
top_k = st.slider("Số kết quả", min_value=1, max_value=50, value=10)

if query:
    results = search(query, vectorizer, tfidf_matrix, df, top_k=top_k)

    if results.empty:
        st.warning("Không tìm thấy kết quả nào.")
    else:
        st.success(f"{len(results)} kết quả")
        for _, row in results.iterrows():
            with st.expander(f"[{row['score']:.4f}]  {row.get('name', '')}"):
                if "category" in row and row["category"]:
                    st.markdown(f"**Danh mục:** {row['category']}")
                if "description" in row and row["description"]:
                    st.markdown(f"**Mô tả:** {row['description']}")
