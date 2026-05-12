import os
import streamlit as st
import torch
import cohere
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, FusionQuery, Fusion, Prefetch

# ===== load env =====
load_dotenv()
QDRANT_URL      = os.getenv("URL_QDRANT")
QDRANT_API_KEY  = os.getenv("API_QDRANT")
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")

COLLECTION_NAME = "Search_Project_Phase0"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# ===== cache resources =====
@st.cache_resource
def load_model():
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())

@st.cache_resource
def load_client():
    return QdrantClient(
        url=require_env("URL_QDRANT"),
        api_key=require_env("API_QDRANT"),
    )

@st.cache_resource
def load_cohere():
    return cohere.Client(require_env("COHERE_API_KEY"))


model  = load_model()
client = load_client()
co     = load_cohere()

# ===== UI =====
st.title("Search san pham (BGE-M3 Hybrid + Cohere Rerank)")
st.caption("Dense + Sparse Lexical Weights → RRF/DBSF boi Qdrant → Rerank boi Cohere")

query      = st.text_input("Nhap ten san pham can tim")
top_k      = st.slider("So ket qua tra ve cuoi cung", 1, 20, 10)
prefetch_k = st.slider("So ket qua lay truoc khi rerank", top_k, 100, min(top_k * 3, 50))
fusion     = st.selectbox("Fusion method", ["RRF", "DBSF"], index=0)
use_rerank = st.checkbox("Bat rerank (Cohere)", value=True)

if query.strip():
    with st.spinner("Dang tim kiem..."):
        # ===== BGE-M3 encode =====
        output = model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vec = output["dense_vecs"][0].tolist()

        lw = output["lexical_weights"][0]
        sparse_vec = SparseVector(
            indices=[int(k) for k in lw.keys()],
            values=[float(v) for v in lw.values()],
        )

        # ===== Qdrant hybrid search =====
        fusion_enum = Fusion.RRF if fusion == "RRF" else Fusion.DBSF

        # Lay nhieu hon neu rerank de co nhieu candidates
        fetch_limit = prefetch_k if use_rerank else top_k

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec,  using="dense",  limit=fetch_limit * 2),
                Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit * 2),
            ],
            query=FusionQuery(fusion=fusion_enum),
            limit=fetch_limit,
            with_payload=True,
        )

        points = results.points

    # ===== Cohere Rerank =====
    if use_rerank and points:
        with st.spinner("Dang rerank voi Cohere..."):
            # Lay ten san pham lam documents de rerank
            documents = [
                pt.payload.get("name", "") for pt in points
            ]

            rerank_response = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=documents,
                top_n=top_k,
            )

            # Sap xep lai points theo thu tu rerank
            reranked_points = []
            for result in rerank_response.results:
                pt = points[result.index]
                reranked_points.append({
                    "point": pt,
                    "rerank_score": result.relevance_score,
                    "original_score": pt.score,
                })

            display_items = reranked_points

    else:
        # Khong rerank, hien thi ket qua goc
        display_items = [
            {"point": pt, "rerank_score": None, "original_score": pt.score}
            for pt in points[:top_k]
        ]

    # ===== Hien thi ket qua =====
    st.write(f"### Ket qua ({len(display_items)} san pham)")

    if not display_items:
        st.info("Khong tim thay san pham phu hop.")
    else:
        for i, item in enumerate(display_items, 1):
            pt            = item["point"]
            name          = pt.payload.get("name", "N/A")
            orig_score    = item["original_score"]
            rerank_score  = item["rerank_score"]

            st.write(f"**{i}. {name}**")

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Qdrant score ({fusion}): {orig_score:.4f}")
            with col2:
                if rerank_score is not None:
                    st.caption(f"Cohere rerank score: {rerank_score:.4f}")

            st.divider()