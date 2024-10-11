from typing import List, Dict
import openai
import logging
import tiktoken

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 128000
CONTEXT_BUFFER = 1000
RESPONSE_WINDOW = 2048
TEMPERATURE = 0.7

BASE_PROMPT = """
You are an AI assistant living in Telegram Group chat. 

Role:
- Summarize the discussion
- Recall incidents
- Provide unbias opinions relevant to the conversation
- Identify task items

User Group:
Expect the user to use both English and Mandarin interchangeably.

Note:
Feel free to process the user requests in step by step order. 
"""


class BaseRAG:
    def __init__(self, vector_collection, openai_api_key: str, embedding_model: str = "text-embedding-3-small", gpt_model: str = "gpt-4o-mini", top_n: int = 20):
        self.vc = vector_collection
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.tokenizer = tiktoken.get_encoding(self.gpt_model)

    def _text_to_embedding(self, text: str):
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.embeddings.create(
                input=text, model=self.embedding_model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"text_to_embedding: {e}")
            raise

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _truncate(self, context: str, max_context_tokens: int) -> str:
        """
        Ensure the total number of tokens in context is less than max_context_tokens.

        Give up older context if necessary.
        """
        truncated_context = []
        total_token_count = 0

        for line in context.split("\n\n"):
            token_count = self._count_tokens(line)
            if total_token_count + token_count <= max_context_tokens:
                truncated_context.append(line)
                total_token_count += token_count
            else:
                break
        return "\n\n".join(truncated_context)

    def retrieve(self, query: str, **kwargs):
        query_embedding = self._text_to_embedding(query)
        relevant_docs = self.vc.query(query_embedding, n_results=self.top_n)
        return "\n\n".join(relevant_docs['documents'][0])

    def augment(self, query: str, context: str, **kwargs):
        context_prompt = f"""
        <context>
        {context}
        </context>
        """

        query_prompt = f"""
        <query>
        {query}
        </query>
        """
        return query_prompt, context_prompt

    def generate(self, query, context, **kwargs):
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": BASE_PROMPT},
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
        context = self._truncate(
            context, CONTEXT_WINDOW - self._count_tokens(query) - CONTEXT_BUFFER)
        query_prompt, context_prompt = self.augment(query, context)
        response = self.generate(query_prompt, context_prompt)
        return response
