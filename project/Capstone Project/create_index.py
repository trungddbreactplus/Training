import os
from dotenv import load_dotenv

from qdrant_client.models import PayloadSchemaType
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API = os.getenv("QDRANT_API")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)

client.create_payload_index(
    collection_name="RAG_ChatBot_HAUI_v1",
    field_name="source",
    field_schema=PayloadSchemaType.KEYWORD
)