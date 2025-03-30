"""

This module manages chatbot memory and related functions.

"""

### Importing libraries ###

import os
import time
import torch

from dotenv                      import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface       import HuggingFaceEmbeddings
from langchain_pinecone          import PineconeVectorStore
from pinecone                    import Pinecone
from pinecone                    import ServerlessSpec

### Environment settings ###

_ = load_dotenv()

EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
INDEX_NAME       = "ybot-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

### Module functions ###

def build_embed_model():
    """
    Build vector embedding model for chatbot LLM.
    """

    embed_model = HuggingFaceEmbeddings(
        encode_kwargs = {"normalize_embeddings" : True},
        model_kwargs  = {"device" : "cpu"},
        model_name    = EMBED_MODEL
    )

    return embed_model

def build_pinecone_vector_store():
    """
    Build Pinecone vector store for chatbot agent.
    """

    # Initialize Pinecone.
    index_name      = INDEX_NAME
    pinecone        = Pinecone(api_key = PINECONE_API_KEY)
    current_indexes = [index_info["name"] for index_info in pinecone.list_indexes()]

    # Build new index if nonexistent.
    if index_name not in current_indexes:
        pinecone.create_index(
            dimension = 384,
            metric    = "cosine",
            name      = index_name,
            spec      = ServerlessSpec(cloud = "aws", region = "us-east-1")
        )

        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)

    pinecone_index = pinecone.Index(index_name)
    embed_model    = build_embed_model()
    vector_store   = PineconeVectorStore(index = pinecone_index, embedding = embed_model)

    return vector_store

def build_vector_store():
    """
    Build LangChain in-memory vector store for chatbot agent.
    """

    embed_model  = build_embed_model()
    vector_store = InMemoryVectorStore(embed_model)

    return vector_store
