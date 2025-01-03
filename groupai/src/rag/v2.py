import json
import logging

import numpy as np  # type: ignore
from llm_agent_toolkit._util import MessageBlock, ResponseMode
from llm_agent_toolkit._memory import VectorMemory, ShortTermMemory
from llm_agent_toolkit._core import Core

from llms import CheckerResponse

logger = logging.getLogger(__name__)


class GroundRAG:
    """
    GroundRAG
    =======

    A Retrieval-Augmented-Generation (RAG) system integrating hybrid memory sources
    for intelligent and context-aware language generation.

    Designed for:
    ----------

    * Chatbot Interactions
    * Knowledge Base Utilization

    Key Features:
    ----------

    * Hybrid Memory Integration:
            * VectorMemory: Semantic Retrieval
            * ShortTermMemory: Recent Interaction

    * Dynamic Retrieval Thresholds:
            * Adapts retrieval filtering based on data density
            using percentile-based distance thresholds for relevance.

    * Context Diversity:
            * File and Chat data are retrieved separately because they are mostly has different distribution.
            Separate the retrieval ensure a diversified and balance context body.

    * Context Management:
            * Token Estimation: Dynamic context window
            * Truncation Strategy: Prioritizes recent data.

    * Composition Design Pattern:
            * vc, encoder, llm, and chat_cache are swappable parts.

    Notes:
    ----------

    * Read-Only Attributes: The `VectorMemory` and `ShortTermMemory` should not be modified.
    * `GroundRAG` is the consumer of the Read-Only attributes.
    """

    def __init__(
        self,
        vector_collection: VectorMemory,
        chat_cache: ShortTermMemory,
        llm: Core,
        checker: Core,
        top_n: int = 20,
    ):
        self.vc = vector_collection
        self.chat_cache = chat_cache
        self.llm = llm
        self.checker = checker
        self.top_n = top_n

    def _estimate_token_count(self, text: str) -> int:
        """Assume 1 token every 2 characters."""
        return len(text) // 2

    def _truncate(
        self, context: list[MessageBlock | dict], max_context_tokens: int
    ) -> list[MessageBlock | dict]:
        """
        Ensure the total number of tokens in context is less than max_context_tokens.

        Notes:
        ------
        - Higher priority to recent content
        - Higher priority to chat content then file content
        """
        context_line = ""
        saturated_at_index = len(context)
        _context = list(ctx for ctx in reversed(context))
        for idx, ctx in enumerate(_context):
            context_line += ctx["content"] + "\n\n"
            if self._estimate_token_count(context_line) >= max_context_tokens:
                saturated_at_index = idx
                break
        return _context[:saturated_at_index]

    def _estimate_context_window(self, query: str) -> int:
        """
        Estimate the remaining context window.
        The estimation is very rough, assume system prompt and tools description use 2K tokens.

        Args:
            query (str): User's query.

        Returns:
            int: The remaining context window.

        Raises:
            ValueError: If the query's exceeded the llm's context length.
        """
        ASSUMPTION: int = 2000
        prompt_token_count: int = self._estimate_token_count(query)
        context_window = self.llm.context_length - prompt_token_count - ASSUMPTION
        if context_window <= 0:
            raise ValueError(
                f"User's query exceed llm's context_length. {prompt_token_count}/{self.llm.context_length}"
            )

        return context_window - self.llm.config.max_output_tokens

    def _retrieve_relevant_content(
        self, query: str, label: str, **kwargs
    ) -> list[tuple[str, str, str, bool]]:
        """Retrieve relevant content of the chosen label based on dynamic threshold.

        Args:
            query (str): User's query.
            label (str): Metadata label.

        Returns:
            List[Tuple[str, str, str, bool, float]]: Relevant content.
        """

        query_response: dict = self.vc.query(
            query_string=query, advance_filter={"label": label}, return_n=self.top_n * 2
        )

        result: dict = query_response["result"]

        ids = result["ids"]
        docs = result["documents"]
        metas = result["metadatas"]
        dists = result["distances"]

        # Dynamic threshold
        if len(ids) > 10:
            max_threshold = np.percentile(dists, 50)
            min_threshold = np.percentile(dists, 10)
        else:
            max_threshold = 2
            min_threshold = 0

        relevant_docs: list[tuple[str, str, str, bool, float]] = []
        for identifier, doc, dist, meta in zip(ids, docs, dists, metas):
            d = (
                identifier,
                f"**{label}**:{doc}",
                meta["lastUpdated"],
                meta["isAnswer"],
                dist,
            )
            if min_threshold <= dist <= max_threshold:
                relevant_docs.append(d)

        if len(relevant_docs) == 0:
            return []

        relevant_docs.sort(key=lambda x: x[-1])  # Sort by distance

        output: list[tuple[str, str, str, bool]] = []
        for idx, doc, ts, is_answer, _ in relevant_docs:
            output.append((idx, doc, ts, is_answer))
            if len(output) == self.top_n:
                break
        return output

    def _retrieve_relevant(self, query: str, **kwargs) -> list[MessageBlock | dict]:
        """
        Retrieving relevant files and chats.

        Args:
            query (str): User's query.

        Returns:
            list[MessageBlock | dict]: Relevant chunks presented in specific dict format (Keys: "role", "content")

        Notes:
        * Expect len(output) between 0 to top_n * 2
        """
        relevant_file_content: list[tuple[str, str, str, bool]] = (
            self._retrieve_relevant_content(query=query, label="file", **kwargs)
        )
        logger.info("Relevant File: %d", len(relevant_file_content))
        relevant_chat_history: list[tuple[str, str, str, bool]] = (
            self._retrieve_relevant_content(query=query, label="chat", **kwargs)
        )
        logger.info("Relevant Chat: %d", len(relevant_chat_history))

        context: list[tuple[str, str, str, bool]] = []
        for chat_hx in relevant_chat_history[:-1]:
            context.append(chat_hx)
        for file_data in relevant_file_content:
            context.append(file_data)

        context.sort(key=lambda x: x[2])  # Sort by timestamp

        output: list[MessageBlock | dict] = []
        for _, line, _, is_answer in context:
            role = "assistant" if is_answer else "user"
            output.append({"role": role, "content": line})

        return output

    def _retrieve_recent(self) -> list[MessageBlock | dict]:
        """Retrieve recent chats."""
        return self.chat_cache.last_n(n=self.top_n)

    def retrieve(
        self, query: str
    ) -> tuple[list[MessageBlock | dict], list[MessageBlock | dict]]:
        """Retrieval Phase: Retrieve relevant and recent data."""
        a = self._retrieve_relevant(query)
        b = self._retrieve_recent()
        return a, b

    def augment(
        self,
        rlv: list[MessageBlock | dict],
        rc: list[MessageBlock | dict],
        max_output_tokens: int,
    ) -> list[MessageBlock | dict]:
        """Augmentation Phase: Mix relevant and recent data into non-duplicated context body.

        Args:
            rlv (List[MessageBlock | dict]): Relevant data
            rc (List[MessageBlock | dict]): Recent data

        Returns:
            List[MessageBlock | dict]
        """
        context: list[MessageBlock | dict] = rlv[:]
        # Detect duplicated content
        for item in rc:
            redundant: bool = False
            for ctx in rlv:
                tmp: str = ctx["content"][:]
                redundant = tmp.strip("**chat**:") == item["content"]
                if redundant:
                    break
            if not redundant:
                context.append(
                    MessageBlock(
                        role=item["role"], content=f"**chat**:{item['content']}"
                    )
                )
        # Trim context to fit the context window
        context = self._truncate(context, max_output_tokens)
        return context

    async def is_grounded(
        self, prompt: str, result: str, context: list[MessageBlock | dict]
    ) -> bool:
        MAX_RETRY = 5
        retry = 0
        challenge: str = f"Query={prompt}\nResponse={result}"
        context_string = ""
        for ctx in context:
            context_string += f"{ctx['content']}\n"

        while retry < MAX_RETRY:
            try:
                responses = await self.checker.run_async(
                    query=challenge,
                    context=[MessageBlock(role="user", content=context_string)],
                    mode=ResponseMode.SO,
                    format=CheckerResponse,
                )

                response = responses[0]
                logger.info("is_grounded: %s", response["content"])
                try:
                    obj = json.loads(response["content"])
                    grounded = obj.get("grounded", False)
                    if not isinstance(grounded, bool):
                        logger.warning("Reason: %s", obj["reason"])
                        return False
                    return grounded
                except json.JSONDecodeError as jde:
                    logger.error("JSONDecodeError: %s", jde)
                return False
            except ValueError as ve:
                if "max_output_tokens <= 0" in str(ve):
                    ori_len = len(context_string)
                    x_len = int(ori_len * 0.8)  # Reduce 20%
                    logger.info("[%d] context_string: %d -> %d", retry, ori_len, x_len)
                    context_string = context_string[:x_len]
                    retry += 1
                else:
                    logger.error("ValueError: %s", ve)
                    raise
        return False

    async def invoke(self, query: str, **kwargs) -> str:
        """Run this method to execute the RAG pipeline."""
        context_window = self._estimate_context_window(query)

        if context_window <= 0:
            raise ValueError("Context window insufficient.")

        # Retrieval Phase
        relevant_context, recent_context = self.retrieve(query)
        # Augmentation Phase
        messages: list[MessageBlock | dict] = self.augment(
            relevant_context, recent_context, context_window
        )

        # Generation Phase
        generated_responses = await self.llm.run_async(query=query, context=messages)

        result_string = ""
        for response in generated_responses:
            result_string += f"{response['content']}\n"

        flag = await self.is_grounded(
            prompt=query, result=result_string, context=messages
        )

        logger.info("%s: %s", flag, result_string)
        if flag:
            return result_string
        return "Not Available"
