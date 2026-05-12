import os
import pandas as pd
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

# ===== load env =====
load_dotenv()
QDRANT_URL     = os.getenv("URL_QDRANT")
QDRANT_API_KEY = os.getenv("API_QDRANT")

COLLECTION_NAME = "Search_Project_Phase0"
CSV_PATH        = "chiaki_products.csv"
BATCH_SIZE      = 16   # BGE-M3 nặng hơn, batch nhỏ để tránh OOM


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# ===== init =====
qdrant = QdrantClient(
    url=require_env("URL_QDRANT"),
    api_key=require_env("API_QDRANT"),
)

print("Loading BGE-M3 ...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=torch.cuda.is_available())

# ===== load & clean data =====
df = pd.read_csv(CSV_PATH)

if "name" not in df.columns:
    raise ValueError("CSV phai co cot 'name'")

df["name"]        = df["name"].fillna("").astype(str)
df["description"] = df["description"].fillna("").astype(str) if "description" in df.columns else ""
df = df[df["name"].str.strip() != ""]
df = df.reset_index(drop=True)

# BGE-M3 ho tro max 8192 token -- ghep name + full description thoai mai
df["text_for_embed"] = df["name"] + ". " + df["description"]

print(f"Tong san pham: {len(df)}")

# ===== create collection =====
collections = [c.name for c in qdrant.get_collections().collections]

if COLLECTION_NAME not in collections:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # BGE-M3 dense dim = 1024
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            # sparse tu BGE-M3 lexical weights (tot hon BM25 thuan)
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            ),
        },
    )
    print(f"Da tao collection: {COLLECTION_NAME}")
else:
    print(f"Collection '{COLLECTION_NAME}' da ton tai, tiep tuc upload ...")

# ===== encode & upload =====
texts = df["text_for_embed"].tolist()
names = df["name"].tolist()
total = len(texts)

for start in range(0, total, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, total)
    batch = texts[start:end]

    # BGE-M3 tra ve dense + sparse (lexical weights) trong 1 lan forward pass
    output = model.encode(
        batch,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs      = output["dense_vecs"]       # shape (batch, 1024)
    lexical_weights = output["lexical_weights"]  # list of dict {token_id: weight}

    points = []
    for i, (name, d_vec, lw) in enumerate(zip(names[start:end], dense_vecs, lexical_weights)):
        indices = [int(k) for k in lw.keys()]
        values  = [float(v) for v in lw.values()]

        points.append(
            PointStruct(
                id=start + i,
                vector={
                    "dense":  d_vec.tolist(),
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload={"name": name},
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Uploaded {end}/{total}")

print("Upload hoan tat!")
