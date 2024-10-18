import os
from typing import List, Dict
import openai
import logging
import tiktoken

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 128000
CONTEXT_BUFFER = 1000
RESPONSE_WINDOW = 2048
TEMPERATURE = 0.7

BASE_PROMPT = os.environ["RAG_PROMPT"]


class BaseRAG:
    def __init__(self, vector_collection, embedding_model: str = "text-embedding-3-small", gpt_model: str = "gpt-4o-mini", top_n: int = 20, initial_instruction: str = ""):
        self.vc = vector_collection
        self.embedding_model = embedding_model
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

    def _truncate(self, context: str, max_context_tokens: int) -> str:
        """
        Ensure the total number of tokens in context is less than max_context_tokens.

        Give up older context if necessary.
        """
        lines = reversed(context.split("\n\n"))
        token_count = self._count_tokens(context, is_embedding=False)
        if token_count <= max_context_tokens:
            return "\n\n".join(lines)
        
        truncated_context = []
        total_token_count = 0
        for line in lines:
            token_count = self._count_tokens(line, is_embedding=False)
            if total_token_count + token_count <= max_context_tokens:
                truncated_context.append(line)
                total_token_count += token_count
            else:
                break
        return "\n\n".join(truncated_context)

    def retrieve_media_context(self, embedding, **kwargs):
        import numpy as np
        media_results = self.vc.query(
            embedding, n_results=self.top_n,
            where={
                "$and":[
                    {"deleted": False},
                    {"isMedia": True}
                ]
            }
        )
        docs = media_results['documents'][0]
        dists = media_results['distances'][0]
        metas = media_results['metadatas'][0]
        max_threshold = np.percentile(dists, 80)
        min_threshold = np.percentile(dists, 20)
        relevant_docs = []
        x = 0
        for doc, dist, meta in zip(docs, dists, metas):
            x += dist
            if min_threshold <= dist <= max_threshold:
                relevant_docs.append(doc)
        if len(relevant_docs) == 0:
            return "No relevant context are found."
        return relevant_docs
    
    def retrieve(self, query: str, threshold: float = 0.5, **kwargs):
        import numpy as np
        query_embedding = self._text_to_embedding(query)
        media_results = self.retrieve_media_context(
            query_embedding, **kwargs
        )
        
        

    def augment(self, query: str, chat_history: str, file_content: str, **kwargs):
        context_prompt = f"""
        <chat_history>
        {chat_history}
        </chat_history>
        
        <file_content>
        {file_content}
        </file_content>
        """

        query_prompt = f"""
        <query>
        {query}
        </query>
        """
        return query_prompt, context_prompt

    def generate(self, query, context, **kwargs):
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": self.initial_instruction},
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            max_tokens=RESPONSE_WINDOW,
            temperature=TEMPERATURE,
            n=1
        )
        return response.choices[0].message.content.strip()

    def __call__(self, query: str, **kwargs):
        context = self.retrieve(query)
        context = self._truncate(context, CONTEXT_WINDOW - self._count_tokens(query) - CONTEXT_BUFFER)
        query_prompt, context_prompt = self.augment(query, context)
        response = self.generate(query_prompt, context_prompt)
        return response
