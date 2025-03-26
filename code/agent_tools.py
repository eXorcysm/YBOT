"""

This module contains the tools called upon by the AI agent.

"""

### Importing libraries ###

import uuid

from typing                   import List
from duckduckgo_search        import DDGS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools     import tool
from .memory                  import build_pinecone_vector_store

### Module functions ###

vector_store = build_pinecone_vector_store()

def get_user_id(config: RunnableConfig) -> str:
    """
    User ID is needed to record memory.
    """

    user_id = config["configurable"].get("user_id")

    if user_id is None:
        raise ValueError("[!] User ID is missing!")

    return user_id

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """
    Save memory to vector store.
    """

    user_id = get_user_id(config)

    doc = Document(
        id           = str(uuid.uuid4()),
        metadata     = {"user_id" : user_id},
        page_content = memory
    )

    vector_store.add_documents([doc])

    return memory

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """
    Search for relevant memories in vector store.
    """

    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    docs = vector_store.similarity_search(
        query,
        filter = _filter_function,
        k      = 3
    )

    return [doc.page_content for doc in docs]

@tool
def search_web(query: str) -> str:
    """
    Search the Internet for user query.
    """

    result = DDGS().text(query, max_results = 3)

    results = [
        {"snippet" : r["body"], "title" : r["title"], "link" : r["href"]} for r in result
    ]

    formatted_results = ""

    for r in results:
        formatted_results += f"Title: {r['title']}\n"
        formatted_results += f"Snippet: {r['snippet']}\n"
        formatted_results += "-----\n"

    return formatted_results
