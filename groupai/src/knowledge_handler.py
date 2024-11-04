import os
import io
import telegram
import tiktoken
import openai
import logging
from typing import List, Optional
from copy import deepcopy
import chromadb

# from model import CompactMessage, Media
# from tlg_msg_scrapper import TlgMsgScraper

logger = logging.getLogger(__name__)


class KnowledgeHandler:
    def __init__(
        self, tmp_directory: str, vdb: chromadb.Client,
        embedding_model: str = "text-embedding-3-small", embedding_chunk_size: int = 4096, stride_rate: float = 0.7,
        gpt_model: str = "gpt-4o-mini", context_window: int = 128000,
    ):
        self.tmp_directory = tmp_directory
        os.makedirs(self.tmp_directory, exist_ok=True)

        self.embedding_model = embedding_model
        self.embedding_chunk_size = embedding_chunk_size
        self.stride = int(embedding_chunk_size * stride_rate)
        # This is not used in current implementation, will be needed when we decided to summary long passages
        self.gpt_model = gpt_model
        self.context_window = context_window
        self.vdb = vdb

    def add(self, namespace: str, identifier: str, knowledge: str, metadata: dict):
        ids, knowledge_chunks, metadatas, embeddings = self.preprocessing(
            identifier, knowledge, metadata)
        collection = self.vdb.get_or_create_collection(
            name=namespace, metadata={
                "hnsw:space": "cosine"
            }
        )
        collection.add(documents=knowledge_chunks,
                       metadatas=metadatas, ids=ids, embeddings=embeddings)

    def update(self, namespace: str, identifier: str, knowledge: str, metadata: dict):
        ids, knowledge_chunks, metadatas, embeddings = self.preprocessing(
            identifier, knowledge, metadata)
        collection = self.vdb.get_or_create_collection(
            name=namespace, metadata={
                "hnsw:space": "cosine"
            }
        )
        collection.update(documents=knowledge_chunks,
                          metadatas=metadatas, ids=ids, embeddings=embeddings)

    def remove(self, namespace: str, method: str, identifier: Optional[str], metadata: Optional[dict]):
        pass

    def preprocessing(self, identifier: str, knowledge: str, metadata: dict):
        ids, metadatas, embeddings = [], [], []
        knowledge_chunks = self.text_to_chunks(knowledge)
        for knowledge_chunk in knowledge_chunks:
            embedding = self.text_to_embedding(knowledge_chunk)
            embeddings.append(embedding)

        if len(knowledge_chunks) == 1:
            ids.append(identifier)
            metadatas.append(metadata)
        else:
            metadatas = []
            for i in range(len(knowledge_chunks)):
                ids.append(f"{identifier}_{i}")
                _meta = deepcopy(metadata)
                _meta["chunk_index"] = i
                metadatas.append(_meta)
        return ids, knowledge_chunks, metadatas, embeddings

    def text_to_embedding(self, text: str):
        # Validation Steps
        assert type(text) == str
        text = text.strip()
        assert len(text) > 0, "Expect non-empty text."
        assert 8192 > len(
            text) > 0, "Expect chunk_size in (0, 8192), but got {}".format(len(text))
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=text, model=self.embedding_model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"transform_text_to_embedding: {e}")
            raise

    def text_to_chunks(self, text: str):
        tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        # token_counts = len(tokenizer.encode(text))
        # if token_counts <= self.embedding_chunk_size:
        #     return [text]

        chunks: List[str] = []
        for start in range(0, len(text), self.stride):
            end = min(start + self.embedding_chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
        return chunks
