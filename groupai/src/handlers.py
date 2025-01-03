import re
import logging
import os
import asyncio
from typing import Any, Optional
import time
from datetime import datetime
import copy

import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode

import chromadb

from llm_agent_toolkit import Encoder
from llm_agent_toolkit.chunkers import SemanticChunker, FixedGroupChunker
from llm_agent_toolkit._memory import ShortTermMemory
from llm_agent_toolkit.memory import ChromaMemory

from model import CompactMessage

from rag.v1 import OneRAG  # type: ignore
from rag.v2 import GroundRAG
from tlg_msg_scrapper import TlgMsgScraper
from myfunction import ChromaDBFactory
import llms

logger = logging.getLogger(__name__)

# Define Global Variables
main_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(
    persist=True, persist_directory="/vect/main"
)
chat_memory: dict[str, ShortTermMemory] = {}

user_locks: dict[str, asyncio.Lock] = {}


def get_user_lock(identifier: str) -> asyncio.Lock:
    """
    Get a lock for a specific user based on their identifier.

    Args:
        identifier (str): The identifier of the user.

    Returns:
        asyncio.Lock: The lock associated with the user.
    """
    global user_locks
    if identifier not in user_locks:
        user_locks[identifier] = asyncio.Lock()
    return user_locks[identifier]


def get_metadata(message: CompactMessage) -> dict[str, Any]:
    """
    Generate metadata for a CompactMessage.

    Args:
        message (CompactMessage): The CompactMessage to generate metadata for.

    Returns:
        dict[str, Any]: A dictionary containing metadata for the CompactMessage.
    """
    metadata = {}
    metadata["created"] = message.created  # datetime string
    metadata["lastUpdated"] = message.lastUpdated  # datetime string
    metadata["username"] = message.username
    metadata["isAnswer"] = message.isAnswer  # bool
    metadata["isForwarded"] = message.isForwarded  # bool
    if message.isForwarded:
        metadata["author"] = message.author
        metadata["isBot"] = message.isBot  # bool
    metadata["edited"] = message.edited  # bool
    metadata["deleted"] = message.deleted  # bool
    metadata["isMedia"] = message.media.isMedia  # bool
    if message.media.isMedia:
        metadata["mime_type"] = message.media.mime_type
    metadata["label"] = "file" if message.media.isMedia else "chat"
    return metadata


async def middleware_function(update: Update, context: CallbackContext) -> None:
    """
    Intercept, process, and store content of incoming Telegram messages, including file upload.

    This asynchronous middleware function handles both new and edited (TODO) messages from Telegram.

    Steps:
    0. Acquires a lock for the user based on their identifier
    1. Parses the message into a CompactMessage format
    2. Converts file upload to markdown format and generate file summary
    3. Stores chat and file (raw text + file summary) in `VectorMemory`
    4. Stores chat in `ShortTermMemory`

    Args:
        update (Update): The incoming update object from Telegram.
        context (CallbackContext): The context object for the current update.

    Returns:
        None

    Raises:
        ValueError: If the message is None.
    """
    global chat_memory, main_vdb
    logger.info("\n>> Middleware Function => %s", update)
    # Extract the message or edited message from the update
    message: Optional[telegram.Message] = getattr(update, "message", None)
    # edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)

    is_edit: bool = False
    if message is None:
        message = getattr(update, "edited_message", None)
        if message is None:
            raise ValueError("Message is None.")
        is_edit = True

    if context.user_data:
        context.user_data["edited_message"] = is_edit

    NAMESPACE = f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    TMP_DIRECTORY = f"/file/{NAMESPACE}"

    ulock = get_user_lock(NAMESPACE)
    async with ulock:
        logger.info("Acquired lock for user: %s", NAMESPACE)

        tlg_msg_scraper = TlgMsgScraper(
            tmp_directory=TMP_DIRECTORY,
            image_interpreter=llms.image_interpreter_llm,
            transcriber=llms.transcriber_llm,
        )
        processed_message: CompactMessage = await tlg_msg_scraper.preprocessing(
            message=message, edited=is_edit
        )

        if processed_message.media.isMedia:
            media_file = await context.bot.get_file(processed_message.media.fileid)
            # Issue: More than one user upload file with the same filename
            TMP_PATH = f"{TMP_DIRECTORY}/{processed_message.media.filename}"
            await media_file.download_to_drive(TMP_PATH)
            processed_message.media.markdown = tlg_msg_scraper.to_markdown(
                processed_message, input_path=TMP_PATH
            )
            os.remove(TMP_PATH)
            if context.user_data:
                context.user_data["media_markdown"] = processed_message.media.markdown

        content = str(processed_message)
        metadata = get_metadata(processed_message)

        if is_edit:
            pass  # vm.remove(processed_message.identifier)

        if processed_message.media.isMedia:
            logger.info("Add file to vector memory...")
            add_file_to_vector_memory(
                namespace=NAMESPACE,
                identifier=processed_message.identifier,
                data=processed_message.media.markdown,
                filename=processed_message.media.filename,
                metadata=metadata,
                encoder=llms.encoder,
            )
            # Create file summary
            file_summary = await llms.summarizer_llm.run_async(
                query="Give me a summary of this file.",
                context=[
                    {
                        "role": "user",
                        "content": content[
                            : llms.summarizer_llm.context_length
                            - llms.summarizer_llm.max_output_tokens
                        ],
                    }
                ],
            )
            metadata["label"] = "summary"
            add_file_to_vector_memory(
                namespace=NAMESPACE,
                identifier=f"{processed_message.identifier}|S",
                data=file_summary,
                filename=processed_message.media.filename,
                metadata=metadata,
                encoder=llms.encoder,
            )

            file_summary = escape_markdown(file_summary)
            await message.reply_text(file_summary, parse_mode=ParseMode.MARKDOWN_V2)
        else:
            logger.info("Add text to vector memory...")
            add_text_to_vector_memory(
                namespace=NAMESPACE,
                identifier=processed_message.identifier,
                data=content,
                metadata=metadata,
                encoder=llms.encoder,
            )

        logger.info("Add chat to chat_memory...")
        if NAMESPACE not in chat_memory:
            logger.info("Register %s to chat_memory.", NAMESPACE)
            chat_memory[NAMESPACE] = ShortTermMemory(max_entry=20)
        chat_memory[NAMESPACE].push({"role": "user", "content": content})


async def error_handler(update: object, context: CallbackContext):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    logger.info("\nError Handler => Update: %s", update)


# pylint: disable-next=unused-argument
async def help_handler(update: Update, context: CallbackContext) -> None:
    if update.message is None:
        raise ValueError("update.message is None")
    await update.message.reply_text(os.environ["REPO_PATH"])


async def message_handler(update: Update, context: CallbackContext) -> None:
    if context.user_data and "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)
    if message is None:
        raise ValueError("Message is None.")
    await message.reply_text("=== COPY ===")


def escape_markdown(text):
    """
    Escape special characters for Telegram's HTMLV2.
    """
    # Characters that need to be escaped
    special_chars = [
        "_",
        # "*",
        "[",
        "]",
        "(",
        ")",
        # "~",
        # "`",
        ">",
        # "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]

    # Escape backslash first to avoid double escaping
    # text = text.replace("\\", "\\\\")
    _text = text[:]
    # # Escape special characters
    for char in special_chars:
        _text = _text.replace(char, f"\\{char}")

    _text = re.sub(r"#{1,} (.*?)\n", r"**\1**\n", _text)
    _text = _text.replace("#", r"\#")
    return _text


def escape_html(text):
    _text = text[:]
    special_characters = [">", "<", "&", "="]
    for c in special_characters:
        _text = _text.replace(c, r"\{c}")
    _text = _text.replace("```python", "```")
    return _text


def compute_k(text_len: int, ctx_length: int) -> int:
    return max(text_len // ctx_length * 2, 1)


def add_text_to_vector_memory(
    namespace: str, identifier: str, data: str, metadata: dict, encoder: Encoder
):
    """
    Add text to vector memory.

    Args:
        namespace (str): Namespace.
        identifier (str): Identifier.
        data (str): Data.
        metadata (dict): Metadata.
        encoder (Encoder): Encoder.

    Returns:
        None
    """
    global main_vdb
    data_len = len(data)
    if data_len >= encoder.ctx_length:
        chunker = SemanticChunker(
            encoder=encoder,
            config={
                "K": compute_k(data_len, encoder.ctx_length),
                "MAX_ITERATION": 20,
                "update_rate": 0.3,
                "min_coverage": 0.9,
            },
        )
    else:
        chunker = FixedGroupChunker(config={"K": 1})

    # Store the knowledge in the knowledge base
    ChromaMemory(
        vdb=main_vdb,
        encoder=encoder,
        chunker=chunker,
        namespace=namespace,
        overwrite=False,
    ).add(
        document_string=data,
        identifier=identifier,
        metadata=metadata,
    )


def add_file_to_vector_memory(
    namespace: str,
    identifier: str,
    data: str,
    filename: str,
    metadata: dict,
    encoder: Encoder,
) -> None:
    """
    Add file content to vector memory.

    Args:
        namespace (str): Namespace.
        identifier (str): Identifier.
        data (str): Data.
        filename (str): Filename.
        metadata (dict): Metadata.
        encoder (Encoder): Encoder.

    Returns:
        None
    """
    global main_vdb
    chunker = SemanticChunker(
        encoder=encoder,
        config={
            "K": compute_k(len(data), encoder.ctx_length),
            "MAX_ITERATION": 50,
            "update_rate": 0.3,
            "min_coverage": 0.9,
        },
    )
    chunks = chunker.split(data)

    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=encoder,
        chunker=FixedGroupChunker(config={"K": 1}),
        namespace=namespace,
        overwrite=False,
    )

    for i, chunk in enumerate(chunks, start=1):
        _metadata = copy.deepcopy(metadata)
        _metadata["page"] = i
        vm.add(
            document_string=f"{filename}:{chunk}",
            identifier=f"{identifier}|{i}",
            metadata=_metadata,
        )


def handle_triple_ticks(text: str, closed: bool):
    TRIPLE_TICKS = "```"
    t = text[:]
    if not closed:
        t = TRIPLE_TICKS + t
    close = len(re.findall(TRIPLE_TICKS, t)) % 2 == 0
    if not close:
        t += TRIPLE_TICKS
    return t, close


async def ask_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, main_vdb, llm

    if context.user_data and "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)

    if message is None:
        raise ValueError("Message is None.")

    NAMESPACE = f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)

    ulock = get_user_lock(NAMESPACE)
    if ulock.locked():
        await message.reply_text("Please wait until the last operation complete.")
        return None

    await message.reply_text("=== PROCESSING... ===")
    # tmp_directory = f"/file/{namespace}"
    chunker = FixedGroupChunker(config={"K": 1})
    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=llms.encoder,
        chunker=chunker,
        namespace=NAMESPACE,
        overwrite=False,
    )
    rag = OneRAG(
        vector_collection=vm,
        chat_cache=chat_memory[NAMESPACE],
        llm=llms.rag_llm,
        top_n=10,
    )
    # Generate the response
    query_text = message.text.strip()
    query_text = query_text.replace("/ask ", "")
    if context.user_data and "media_markdown" in context.user_data:
        query_text += f"\n\n{context.user_data['media_markdown']}"

    async with ulock:
        # answer = rag(query=query_text, user_identifier=namespace)
        answer = await rag.invoke(query=query_text, user_identifier=NAMESPACE)
        chat_memory[NAMESPACE].push({"role": "assistant", "content": answer})
        # answer = escape_markdown(answer)
        # Send the response
        logger.info("Answer: %s", answer)

        if len(answer) >= 4096:
            sections = re.split(r"\n{2,}", answer)
            current_chunk = ""
            is_close: bool = True
            for section in sections:
                if len(current_chunk) + len(section) + 2 <= 4096:
                    current_chunk += section + "\n\n"
                else:
                    current_chunk = escape_markdown(current_chunk)
                    current_chunk, is_close = handle_triple_ticks(
                        current_chunk, is_close
                    )
                    _ = await message.reply_text(
                        current_chunk, parse_mode=ParseMode.MARKDOWN_V2
                    )
                    current_chunk = ""
            # Handle the last chunk.
            if current_chunk:
                current_chunk, is_close = handle_triple_ticks(current_chunk, is_close)
                _ = await message.reply_text(
                    current_chunk, parse_mode=ParseMode.MARKDOWN_V2
                )
        else:
            answer = escape_markdown(answer)
            _ = await message.reply_text(answer, parse_mode=ParseMode.MARKDOWN_V2)

        identifier = f"{message.chat.id}/{message.message_id}/r{int(time.time())}"
        ts = str(datetime.now())
        metadata = {
            "username": llms.rag_llm.model_name,
            "isAnswer": True,
            "created": ts,
            "lastUpdated": ts,
            "label": "chat",
        }
        add_text_to_vector_memory(
            NAMESPACE,
            identifier,
            answer,
            metadata,
            llms.encoder,
        )


async def summary_handler(update: Update, context: CallbackContext) -> None:
    global chat_memory, main_vdb, encoder

    if context.user_data and "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)
    if message is None:
        raise ValueError("Message is None.")

    NAMESPACE = f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)

    ulock = get_user_lock(NAMESPACE)
    if ulock.locked():
        await message.reply_text("Please wait until the last operation complete.")
        return None

    await message.reply_text("=== PROCESSING... ===")
    # tmp_directory = f"/file/{namespace}"
    chunker = FixedGroupChunker(config={"K": 1})
    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=llms.encoder,
        chunker=chunker,
        namespace=NAMESPACE,
        overwrite=False,
    )
    rag = GroundRAG(
        vector_collection=vm,
        chat_cache=chat_memory[NAMESPACE],
        llm=llms.summarizer_llm,
        checker=llms.checker_llm,
        top_n=30,
    )
    logger.info("GroundRAG: %s", rag)
    query_text: str = message.text.strip()
    query_text = query_text.replace("/summary ", "")

    if query_text:
        query_text = "Topic/Keyword: General Summary"
    else:
        query_text = f"Topic/Keyword: {query_text}"

    # if "media_markdown" in context.user_data:
    # query_text += f"\n\nMedia: {context.user_data['media_markdown']}"

    async with ulock:
        answer = await rag.invoke(query=query_text, user_identifier=NAMESPACE)
        chat_memory[NAMESPACE].push({"role": "assistant", "content": answer})
        # answer = escape_markdown(answer)
        if len(answer) >= 4096:
            sections = re.split(r"\n{2,}", answer)
            current_chunk = ""
            is_close: bool = True
            for section in sections:
                if len(current_chunk) + len(section) + 2 <= 4096:
                    current_chunk += section + "\n\n"
                else:
                    current_chunk = escape_markdown(current_chunk)
                    current_chunk, is_close = handle_triple_ticks(
                        current_chunk, is_close
                    )
                    _ = await message.reply_text(
                        current_chunk, parse_mode=ParseMode.MARKDOWN_V2
                    )
                    current_chunk = ""
            # Handle the last chunk.
            if current_chunk:
                current_chunk = escape_markdown(current_chunk)
                current_chunk, is_close = handle_triple_ticks(current_chunk, is_close)
                _ = await message.reply_text(
                    current_chunk, parse_mode=ParseMode.MARKDOWN_V2
                )
        else:
            answer = escape_markdown(answer)
            _ = await message.reply_text(answer, parse_mode=ParseMode.MARKDOWN_V2)

        identifier = f"{message.chat.id}/{message.message_id}/r{int(time.time())}"
        ts = str(datetime.now())
        metadata = {
            "username": llms.summarizer_llm.model_name,
            "isAnswer": True,
            "created": ts,
            "lastUpdated": ts,
            "label": "chat",
        }
        add_text_to_vector_memory(NAMESPACE, identifier, answer, metadata, llms.encoder)


# async def export_handler(update: Update, context: CallbackContext) -> None:
#     """
#     Export chat history in csv format.

#     Retrieve chat history from the specified chat through `forward_message`.
#     Can choose to export only recent messages or all messages.
#     Due to the limitation of Telegram API, the bot will not be notified when a message is deleted.
#     Therefore, this program iteratively challenges the existence of a message.

#     Along this time, the program will also try to retrieve previously uncought messages either due to bot downtime or lost of .db file.

#     This function retrieves the chat history from a specified chat using `forward_message` method.
#     It allows the user to export either only recent messages or the entire chat history.
#     Due to Telegram API limitations, the bot cannot detect when a message is deleted directly;
#     thus, the program iteratively checks the existence of each message to handle deletions.

#     Additionally, this function attempts to retrieve any messages that were missed,
#     potentially due to bot downtime or the loss of the database file.

#     Limitations:
#     - When forwarding messages, the Telegram API changes `msg.chat` to represent the bot
#       and `msg.from_user` to represent the bot's user, masking the original sender's identity.
#     - In group chats, forwarding a message does not capture the identity of the user who forwarded it;
#       instead, it displays the original sender and the bot.

#     Side Effects:
#     - This function forward messages to a master chat to verify their existence.
#     - It stores chat messages in an SQLite database for persistent storage.

#     Note:
#     - This function challenges the existence of messages by attempting to forward them.
#     - It may mark messages as deleted if the forwarding fails.
#     """
#     # Configuration
#     recent: bool = False

#     caller_name = (
#         update.message.from_user.username
#         or f"{update.message.from_user.first_name} {update.message.from_user.last_name}"
#     )
#     chatid = update.message.chat.id
#     storage = SQLite3_Storage(f"/file/{chatid}.db", overwrite=False)

#     messageid = update.message.message_id
#     chattype = update.message.chat.type
#     chatname = update.message.chat.title or (
#         f"{update.message.chat.first_name} {update.message.chat.last_name}"
#     )
#     # Determine search range
#     if recent:
#         search_from = messageid - 20
#     else:
#         search_from = 0
#     search_to = messageid
#     for i in range(search_from, search_to):
#         try:
#             key = f"{chatid}/{i}"
#             result = storage.get(key)
#             # Challenge the existence of a message
#             msg = await context.bot.forward_message(
#                 chat_id=master,
#                 message_id=i,
#                 from_chat_id=chatid,
#                 disable_notification=True,
#             )
#             if result is None:
#                 if (
#                         msg.forward_origin.type
#                         is telegram.constants.MessageOriginType.HIDDEN_USER
#                 ):
#                     forward_origin: telegram.MessageOriginHiddenUser = msg.forward_origin
#                     forward_sender_name = forward_origin.sender_user_name
#                     is_bot = False
#                 else:
#                     forward_origin: telegram.MessageOriginUser = msg.forward_origin
#                     forward_sender_name = (
#                         f"{forward_origin.sender_user.first_name} {forward_origin.sender_user.last_name}"
#                         or forward_origin.sender_user.username
#                     )
#                     is_bot = forward_origin.sender_user.is_bot

#                 if (
#                         update.message.chat.type is telegram.constants.ChatType.PRIVATE
#                         and forward_sender_name != caller_name
#                 ):
#                     is_forwarded = True
#                 else:
#                     # forward_sender_name[-3:].lower() == "bot":
#                     is_forwarded = False

#                 # Set username and userid as None since we cannot discern it's original sender.
#                 # To be honest, we do not know the original created datetime
#                 result = CompactMessage(
#                     identifier=key,
#                     text=msg.text or msg.caption,
#                     chattype=chattype,
#                     chatid=chatid,
#                     chatname=chatname,
#                     userid=None,
#                     username=None,
#                     message_id=i,
#                     created=None,
#                     lastUpdated=str(msg.forward_origin.date),
#                     edited=False,
#                     deleted=False,
#                     isForwarded=is_forwarded,
#                     author=forward_sender_name,
#                     isBot=is_bot,
#                     media=await myfunction.extract_media(msg),
#                 )
#                 storage.set(key, result.to_dict())
#         except telegram.error.BadRequest as bad_request:
#             if result:
#                 # Message has been deleted
#                 result["deleted"] = True
#                 storage.set(key, result)
#                 logger.error(f"Failed to copy message({key}): {bad_request}")
#             else:
#                 logger.error(f"Failed to copy message({key}): {bad_request}")

#     if update.message.chat.title:
#         export_path = f"/file/{update.message.chat.title}_{int(time())}.csv"
#     else:
#         export_path = f"/file/{update.message.chat.id}_{int(time())}.csv"
#     storage.export_csv(export_path)
#     reply_msg = await update.message.reply_document(
#         export_path, parse_mode=ParseMode.HTML
#     )
#     conversation = CompactMessage(
#         identifier=f"{reply_msg.chat.id}/{reply_msg.message_id}",
#         text=None,
#         chattype=reply_msg.chat.type,
#         chatid=reply_msg.chat.id,
#         chatname=reply_msg.chat.title or f"{reply_msg.chat.first_name} {reply_msg.chat.last_name}",
#         userid=reply_msg.from_user.id,
#         username=reply_msg.from_user.username,
#         message_id=reply_msg.message_id,
#         created=str(reply_msg.date),
#         lastUpdated=str(reply_msg.date),
#         edited=False,
#         deleted=False,
#         isForwarded=False,
#         author=None,
#         isBot=False,
#         media=await myfunction.extract_media(reply_msg),
#     )
#     storage = SQLite3_Storage(
#         f"/file/{conversation.chatid}.db", overwrite=False)
#     storage.set(conversation.identifier, conversation.to_dict())
