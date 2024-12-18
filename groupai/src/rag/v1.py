import os
import logging

# import tiktoken
import numpy as np  # type: ignore
from llm_agent_toolkit._memory import VectorMemory, ShortTermMemory
from llm_agent_toolkit._core import Core
from llm_agent_toolkit._encoder import Encoder

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 128000
CONTEXT_BUFFER = 1000
RESPONSE_WINDOW = 2048
TEMPERATURE = 0.7

BASE_PROMPT = os.environ["RAG_PROMPT"]


class OneRAG:
    """OneRAG"""

    def __init__(
        self,
        vector_collection: VectorMemory,
        chat_cache: ShortTermMemory,
        encoder: Encoder,
        llm: Core,
        top_n: int = 20,
    ):
        self.vc = vector_collection
        self.encoder = encoder
        self.llm = llm
        self.threshold = 0.5  # This is highly dependent on the embedding model!
        self.top_n = top_n
        self.chat_cache = chat_cache

    def _text_to_embedding(self, text: str) -> list[float]:
        try:
            response = self.encoder.encode(text=text)
            return response
        except Exception as e:
            logger.error("text_to_embedding: Error=%s", str(e))
            raise

    def _count_tokens(self, text: str) -> int:
        """Assume 1 token/character."""
        return len(text)

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

        token_count = self._count_tokens(context_line)
        if token_count <= max_context_tokens:
            return context

        total_token_count = 0
        selected_chat: list[tuple[bool, str]] = []
        for role, line in reversed(context):
            token_count = self._count_tokens(line)
            if total_token_count + token_count <= max_context_tokens:
                selected_chat.insert(0, (role, line))
                total_token_count += token_count
            else:
                break
        return selected_chat

    def retrieve_relevant_content(
        self, query: str, label: str, **kwargs
    ) -> list[tuple[str, str, str, bool]]:
        """Retrieve relevant content."""
        # Dynamic threshold
        query_response: dict = self.vc.query(
            query_string=query, advance_filter={"label": label},
        )
        result: dict = query_response["result"]
        ids = result["ids"]
        docs = result["document"]
        metas = result["metadata"]
        dists = result["distance"]
        if len(ids) > self.top_n:
            max_threshold = np.percentile(dists, 50)
            min_threshold = np.percentile(dists, 20)
        else:
            max_threshold = 2
            min_threshold = 0
        relevant_docs: list[tuple[str, str, str, bool]] = []
        x = 0
        for identifier, doc, dist, meta in zip(ids, docs, dists, metas):
            x += dist
            d = (identifier, f"**{label}**:{doc}", meta["lastUpdated"], meta["isAnswer"])
            if min_threshold <= dist <= max_threshold:
                # Whether to polish the doc with metadata
                relevant_docs.append(d)
            elif dist < self.threshold:
                relevant_docs.append(d)
        if len(relevant_docs) == 0:
            return relevant_docs
        relevant_docs.sort(key=lambda x: x[2])
        return relevant_docs[:self.top_n]

    def retrieve(self, query: str, **kwargs) -> list[tuple[bool, str]]:
        """Retrieving..."""
        relevant_file_content: list[tuple[str, str, str, bool]] = (
            self.retrieve_relevant_content(query=query, label="file", **kwargs)
        )
        logger.info(f"Relevant File: %d", len(relevant_file_content))
        relevant_chat_history: list[tuple[str, str, str, bool]] = (
            self.retrieve_relevant_content(query=query, label="chat", **kwargs)
        )
        logger.info("Relevant Chat: %d", len(relevant_chat_history))
        context: list[tuple[str, str, str, bool]] = []
        for chat_hx in relevant_chat_history[:-1]:
            context.append(chat_hx)
        for file_data in relevant_file_content:
            context.append(file_data)
        context.sort(key=lambda x: x[2])
        return [(role, line) for (id, line, ts, role) in context]

    def augment(self, history: list[tuple[bool, str]], **kwargs) -> list:
        """Bundle history -> Context"""
        messages = [
            {"role": "system", "content": BASE_PROMPT},
        ]
        for is_ai, line in history:
            role = "assistant" if is_ai else "user"
            messages.append({"role": role, "content": line})
        return messages

    def generate(self, query: str, messages: list, **kwargs):
        """Call the LLM."""
        generated_responses = self.llm.run(query=query, context=messages)
        result_string = ""
        for response in generated_responses:
            result_string += f"{response['content']}\n"
        return result_string

    def __call__(self, query: str, user_identifier: str, **kwargs):
        max_context_tokens = (
            CONTEXT_WINDOW
            - self._count_tokens(query)
            - CONTEXT_BUFFER
            - RESPONSE_WINDOW
        )  # This is not accurate!
        relevant_context: list[tuple[bool, str]] = self.retrieve(query)

        context = self._truncate(relevant_context, max_context_tokens)
        messages: list = self.augment(context)
        recent_context: list[dict] = self.chat_cache.last_n(n=self.top_n)
        if recent_context:
            messages.extend(recent_context)
        response = self.generate(query, messages)

        self.chat_cache.push({"role": "user", "content": query})
        self.chat_cache.push({"role": "assistant", "content": response})
        return response

    async def invoke(self, query: str, user_identifier: str, **kwargs):
        max_context_tokens = (
            CONTEXT_WINDOW
            - self._count_tokens(query)
            - CONTEXT_BUFFER
            - RESPONSE_WINDOW
        )  # This is not accurate!
        relevant_context: list[tuple[bool, str]] = self.retrieve(query)

        context = self._truncate(relevant_context, max_context_tokens)
        messages: list = self.augment(context)
        recent_context: list[dict] = self.chat_cache.last_n(n=self.top_n)
        if recent_context:
            messages.extend(recent_context)
        # response = self.generate(query, messages)
        generated_responses = await self.llm.run_async(query=query, context=messages)
        result_string = ""
        for response in generated_responses:
            result_string += f"{response['content']}\n"

        self.chat_cache.push({"role": "user", "content": query})
        self.chat_cache.push({"role": "assistant", "content": result_string})
        return result_string
