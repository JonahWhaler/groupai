import os
import json
import chromadb
import openai
from typing import Optional
import numpy as np

FILE_EXPLORER_AGENT_PROMPT = """
Description: Searching and analyzing content from the local vector database of documents and files.

Parameters:

`query` (string): The query to be executed on DuckDuckGo. Constraint: 1 <= len(query) <= 200
`namespace` (string): The namespace to search in.
`top_n` (int): The number of results to return. Default is 5. Constraint: 1 <= top_n <= 20

Returns:

`response` (dict): {"result": [({ID}, {DOC}, {LAST_UPDATED}, {IS_ANSWER})]}
`next_func` (string): The name of the next function to call.

**Keywords:** document, file, database, vector, embedding, storage, local, internal, archive, repository, collection, folder, directory, record, content, passage, section, page, chapter, text, stored, saved, existing, indexed

**Relevant Query Types:**
- "Find documents about..."
- "Search our files for..."
- "Look up internal information about..."
- "Get stored content related to..."
- "Find passages mentioning..."
"""


class FileExplorerAgent:
    def __init__(
        self, priority: int = 1, next_func: Optional[str] = None,
        vector_database: Optional[chromadb.ClientAPI] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.__name = "FileExplorerAgent"
        self.__priority = priority
        self.__next_func = next_func
        self.__vdb = vector_database
        self.__embedding_model = embedding_model

    @property
    def priority(self):
        return self.__priority

    @property
    def prompt(self):
        return FILE_EXPLORER_AGENT_PROMPT
    
    @classmethod
    def validate(cls, params: str) -> bool:
        params = json.loads(params)
        namespace: str = params.get("namespace", None)
        query: str = params.get("query", None)
        top_n = params.get("top_n", 5)
        if query is None or namespace is None:
            return False
        query = query.strip()
        condition = [len(query) > 0, len(query) <= 200, top_n >= 1, top_n <= 20]
        if not all(condition):
            return False
        return True

    def _text_to_embedding(self, text: str):
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=text, model=self.__embedding_model)
            return response.data[0].embedding
        except Exception as e:
            # logger.error(f"text_to_embedding: {e}")
            raise

    async def __call__(self, params: str) -> tuple[dict, str | None]:
        params = json.loads(params)
        namespace = params.get("namespace", None)
        query = params.get("query", None)
        top_n = params.get("top_n", 20)
        print(f"Begin Execution: {self.__name}...")
        v_collection = self.__vdb.get_or_create_collection(
            name=namespace, metadata={"hnsw:space": "cosine"}
        )
        output = dict()
        if self.__vdb is not None:
            query_embedding = self._text_to_embedding(query)
            relevant_docs = v_collection.query(query_embedding, n_results=top_n,
                                               where={
                                                   "$and": [
                                                       {"deleted": False},
                                                       {"isMedia": True}
                                                   ]
                                               },
                                               include=['metadatas', 'documents', 'distances'])
        ids = relevant_docs['ids'][0]
        docs = relevant_docs['documents'][0]
        dists = relevant_docs['distances'][0]
        metas = relevant_docs['metadatas'][0]
        if len(ids) >= 5:
            max_threshold = np.percentile(dists, 50)
            min_threshold = np.percentile(dists, 20)
        else:
            max_threshold = 2
            min_threshold = 0
        relevant_docs: list[tuple[str, str, str, bool]] = []
        x = 0
        for id, doc, dist, meta in zip(ids, docs, dists, metas):
            x += dist
            d = (id, doc, meta["lastUpdated"], False)
            if min_threshold <= dist <= max_threshold:
                # Whether to polish the doc with metadata
                relevant_docs.append(d)
            elif dist < self.threshold:
                relevant_docs.append(d)
        if len(relevant_docs) == 0:
            return relevant_docs
        relevant_docs.sort(key=lambda x: x[2])
        print(f"End Execution: {self.__name}")
        output["result"] = relevant_docs
        return output, self.__next_func
