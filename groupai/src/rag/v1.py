import os
from typing import List, Dict, Tuple
import openai
import logging
import tiktoken
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 128000
CONTEXT_BUFFER = 1000
RESPONSE_WINDOW = 2048
TEMPERATURE = 0.7

BASE_PROMPT = os.environ["RAG_PROMPT"]


class OneRAG:
    def __init__(self, vector_collection: Collection, embedding_model: str = "text-embedding-3-small", gpt_model: str = "gpt-4o-mini", top_n: int = 20, initial_instruction: str = ""):
        self.vc = vector_collection
        self.embedding_model = embedding_model
        self.threshold = 0.5 # This is highly dependent on the embedding model!
        self.gpt_model = gpt_model
        self.top_n = top_n
        if initial_instruction == "" or initial_instruction is None:
            self.initial_instruction = BASE_PROMPT
        else:
            self.initial_instruction = initial_instruction

    def _text_to_embedding(self, text: str):
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=text, model=self.embedding_model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"text_to_embedding: {e}")
            raise

    def _count_tokens(self, text: str, is_embedding: bool = True) -> int:
        if is_embedding:
            tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        else:
            tokenizer = tiktoken.encoding_for_model(self.gpt_model)
        return len(tokenizer.encode(text))

    def _truncate(
        self, context: list[tuple[bool, str]], max_context_tokens: int
    ) -> list[tuple[bool, str]]:
        """
        Ensure the total number of tokens in context is less than max_context_tokens.

        Notes:
        ------
        - Higher priority to recent content
        - Higher priority to chat content then file content
        """
        context_line = "\n\n".join([line[1] for line in context])
        
        token_count = self._count_tokens(context_line, is_embedding=False)
        if token_count <= max_context_tokens:
            return context

        total_token_count = 0
        selected_chat: list[tuple[bool, str]] = []
        for (role, line) in reversed(context):
            token_count = self._count_tokens(line, is_embedding=False)
            if total_token_count + token_count <= max_context_tokens:
                selected_chat.insert(0, (role, line))
                total_token_count += token_count
            else:
                break
        return selected_chat

    def retrieve_relevant_file_content(self, embedding, **kwargs) -> list[tuple[str, str, str, bool]]:
        # Dynamic threshold
        import numpy as np
        media_results = self.vc.query(
            embedding, n_results=self.top_n,
            where={
                "$and": [
                    {"deleted": False},
                    {"isMedia": True}
                ]
            },
            include=['metadatas', 'documents', 'distances']
        )
        ids = media_results['ids'][0]
        docs = media_results['documents'][0]
        dists = media_results['distances'][0]
        metas = media_results['metadatas'][0]
        max_threshold = np.percentile(dists, 80)
        min_threshold = np.percentile(dists, 20)
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
        return relevant_docs

    def retrieve_most_recent_chat_history(self, ) -> list[tuple[str, str, str, bool]]:
        result = self.vc.get(
            where={
                "$and": [
                    {"deleted": False},
                    {"isMedia": False}
                ]
            },
            include=['metadatas', 'documents']
        )
        ids = result['ids']
        docs = result['documents']
        metas = result['metadatas']
        chat_history = []
        for id, doc, meta in zip(ids, docs, metas):
            logger.info(f"\n>>> id: {id}, doc: {doc}, meta: {meta}")
            chat = (id, doc, meta["lastUpdated"], meta["isAnswer"])
            chat_history.append(chat)
        if len(chat_history) == 0:
            return chat_history
        chat_history.sort(key=lambda x: x[2])
        return chat_history[-self.top_n:]

    def retrieve_relevant_chat_history(self, embedding, **kwargs) -> list[tuple[str, str, str, bool]]:
        # Dynamic threshold
        import numpy as np
        result = self.vc.query(
            embedding, n_results=self.top_n,
            where={
                "$and": [
                    {"deleted": False},
                    {"isMedia": False}
                ]
            },
            include=['metadatas', 'documents', 'distances']
        )
        ids = result['ids'][0]
        docs = result['documents'][0]
        dists = result['distances'][0]
        metas = result['metadatas'][0]
        max_threshold = np.percentile(dists, 80)
        min_threshold = np.percentile(dists, 20)
        chat_history = []
        x = 0
        for id, doc, dist, meta in zip(ids, docs, dists, metas):
            x += dist
            if min_threshold <= dist <= max_threshold:
                chat = (id, doc, meta["lastUpdated"], meta["isAnswer"])
                chat_history.append(chat)
            elif dist < self.threshold:
                chat = (id, doc, meta["lastUpdated"], meta["isAnswer"])
                chat_history.append(chat)
        if len(chat_history) == 0:
            return chat_history
        chat_history.sort(key=lambda x: x[2])
        return chat_history

    def retrieve_chat_history(self, query_embedding, **kwargs) -> list[tuple[str, str, str, bool]]:
        relevant_chat_history: list[tuple[str, str, str, bool]] = self.retrieve_relevant_chat_history(
            query_embedding, **kwargs)
        recent_chat_history: list[tuple[str, str, str, bool]] = self.retrieve_most_recent_chat_history()
        included_ids = set()
        chat_history = []
        for chat in recent_chat_history:
            included_ids.add(chat[0])
            chat_history.append(chat)
        for chat in relevant_chat_history:
            if chat[0] in included_ids:
                continue
            chat_history.append(chat)
        chat_history.sort(key=lambda x: x[2])
        return chat_history
    
    def retrieve(self, query: str, **kwargs) -> list[tuple[bool, str]]:
        query_embedding = self._text_to_embedding(query)
        chat_history: list[tuple[str, str, str, bool]] = self.retrieve_chat_history(query_embedding, **kwargs)
        relevant_file_content: list[tuple[str, str, str, bool]] = self.retrieve_relevant_file_content(
            query_embedding, **kwargs
        )
        context = []
        for chat in chat_history:
            context.append(chat)
        for file in relevant_file_content:
            context.append(file)
        context.sort(key=lambda x: x[2])
        return [(role, line) for (id, line, ts, role) in context]

    def augment(self, query: str, history: list[tuple[bool, str]], **kwargs) -> List:
        messages = [
            {"role": "system", "content": self.initial_instruction},
        ]
        for (isAI, line) in history:
            role = "assistant" if isAI else "user"
            messages.append({"role": role, "content": line})
        messages.append({"role": "user", "content": query})
        return messages

    def generate(self, messages: list, **kwargs):
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.gpt_model,
            messages=messages,
            max_tokens=RESPONSE_WINDOW,
            temperature=TEMPERATURE,
            n=1
        )
        return response.choices[0].message.content.strip()

    def __call__(self, query: str, **kwargs):
        max_context_tokens = CONTEXT_WINDOW - self._count_tokens(query) - CONTEXT_BUFFER - RESPONSE_WINDOW
        context: list[tuple[bool, str]] = self.retrieve(query)
        context = self._truncate(context, max_context_tokens)
        messages: list = self.augment(query, context)
        response = self.generate(messages)
        return response
