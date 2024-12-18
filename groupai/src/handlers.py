import re
import logging
import os
from typing import Optional
import time
from datetime import datetime

import telegram
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode

import chromadb

from llm_agent_toolkit.core import local, open_ai
from llm_agent_toolkit.encoder.remote import OpenAIEncoder
from llm_agent_toolkit.encoder.local import OllamaEncoder
from llm_agent_toolkit.chunkers import SemanticChunker, FixedGroupChunker
from llm_agent_toolkit._memory import ShortTermMemory
from llm_agent_toolkit.memory import ChromaMemory
from llm_agent_toolkit import ChatCompletionConfig, ImageGenerator, Transcriber, TranscriptionConfig, Core
from llm_agent_toolkit.transcriber.open_ai import OpenAITranscriber

import config
from model import CompactMessage

from rag.v1 import OneRAG  # type: ignore
from tlg_msg_scrapper import TlgMsgScraper
from myfunction import ChromaDBFactory

logger = logging.getLogger(__name__)

# Define Global Variables
local.OllamaCore.load_csv("/files/ollama.csv")
encoder = OllamaEncoder(
    connection_string=config.CONNECTION_STRING, model_name=config.emb_model_name
)
llm: Core = local.Text_to_Text(connection_string=config.CONNECTION_STRING, **config.main_t2t_config)
ii: ImageGenerator = local.Image_to_Text(connection_string=config.CONNECTION_STRING, **config.main_i2t_config)
transcriber: Transcriber = OpenAITranscriber(TranscriptionConfig(name=config.a2t_model_name))
main_vdb: chromadb.ClientAPI = ChromaDBFactory.get_instance(persist=True, persist_directory="/vect/main")
chat_memory: dict[str, ShortTermMemory] = {}


def get_metadata(message: CompactMessage) -> dict:
    metadata = {}
    metadata["created"] = message.created
    metadata["lastUpdated"] = message.lastUpdated
    metadata["username"] = message.username
    metadata["isAnswer"] = message.isAnswer
    metadata["isForwarded"] = message.isForwarded
    if message.isForwarded:
        metadata["author"] = message.author
        metadata["isBot"] = message.isBot
    metadata["edited"] = message.edited
    metadata["deleted"] = message.deleted
    metadata["isMedia"] = message.media.isMedia
    if message.media.isMedia:
        metadata["mime_type"] = message.media.mime_type
    metadata["label"] = "file" if message.media.isMedia else "chat"
    return metadata


async def middleware_function(update: Update, context: CallbackContext) -> None:
    """
    Intercept, process, and store content of incoming Telegram messages, including media, in databases.

    This asynchronous middleware function handles both new and edited messages from Telegram.
    It parses the message into a CompactMessage format, processes any media content,
    generates text embeddings, and stores the information in both SQLite and vector databases.

    Args:
        update (Update): The incoming update object from Telegram.
        context (CallbackContext): The context object for the current update.

    Returns:
        None

    Raises:
        No exceptions are explicitly raised, but errors are logged.

    """
    global chat_memory, main_vdb
    logger.info("\nMiddleware Function => Update: %s", update)
    # Extract the message or edited message from the update
    message: Optional[telegram.Message] = getattr(update, "message", None)
    edited_message: Optional[telegram.Message] = getattr(update, "edited_message", None)
    if not message and not edited_message:
        logger.error("\nException: [Message Body Not Found]=> Update: %s", update)
        return None

    if edited_message and context.user_data:
        context.user_data["edited_message"] = True
        namespace = (
            f"g{edited_message.chat.id}"
            if edited_message.chat.id < 0
            else str(edited_message.chat.id)
        )
    elif message:
        namespace = (
            f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
        )
    else:
        raise ValueError("Message is None.")

    tmp_directory = f"/file/{namespace}"
    tlg_msg_scraper = TlgMsgScraper(
        tmp_directory=tmp_directory, image_interpreter=ii, transcriber=transcriber
    )
    if edited_message:
        processed_message = await tlg_msg_scraper.preprocessing(edited_message, True)
    else:
        processed_message = await tlg_msg_scraper.preprocessing(message, False)

    if processed_message.media.isMedia:
        media_file = await context.bot.get_file(processed_message.media.fileid)
        # Issue: More than one user upload file with the same filename
        tmp_path = f"{tmp_directory}/{processed_message.media.filename}"
        await media_file.download_to_drive(tmp_path)
        processed_message.media.markdown = tlg_msg_scraper.to_markdown(
            processed_message, input_path=tmp_path
        )
        os.remove(tmp_path)
        if context.user_data:
            context.user_data["media_markdown"] = processed_message.media.markdown
        else:
            logger.warning("context.user_data is None.")

    metadata = get_metadata(processed_message)
    # logger.info(f"Metadat: {metadata}")
    content = str(processed_message)
    logger.info("Information: %s", content)
    K = max(len(content) // encoder.ctx_length * 2, 1)
    if processed_message.media.isMedia and K > 1:
        chunker = SemanticChunker(
            encoder=encoder,
            config={
                "K": K,
                "MAX_ITERATION": 50,
                "update_rate": 0.3,
                "min_coverage": 0.9,
            },
        )
    else:
        chunker = FixedGroupChunker(
            config={
                "K": max(K, 1)
            }
        )
    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=encoder,
        chunker=chunker,
        namespace=namespace,
        overwrite=False,
    )
    # if "edited_message" in context.user_data:
    #     vm.update
    vm.add(
        document_string=content,
        identifier=processed_message.identifier,
        metadata=metadata,
    )
    if namespace not in chat_memory:
        logger.info("Register %s to chat_memory.", namespace)
        chat_memory[namespace] = ShortTermMemory(max_entry=20)
    chat_memory[namespace].push({"role": "user", "content": content})


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
    import re
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
        _text = _text.replace(c, f"\{c}")
    _text = _text.replace("```python", "```")
    return _text

def add_to_vector_memory(namespace: str, identifier: str, data: str, metadata: dict):
    global chat_memory, main_vdb
    if len(data) >= encoder.ctx_length:
        chunker = SemanticChunker(
            encoder=encoder,
            config={
                "K": len(data) // encoder.ctx_length,
                "MAX_ITERATION": 20,
                "update_rate": 0.3,
                "min_coverage": 0.9,
            },
        )
    else:
        chunker = FixedGroupChunker(config={"K": 1})

    # Store the knowledge in the knowledge base
    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=encoder,
        chunker=chunker,
        namespace=namespace,
        overwrite=False,
    )
    vm.add(
        document_string=data,
        identifier=identifier,
        metadata=metadata,
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
    global chat_memory, main_vdb, encoder, llm
    if context.user_data and "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)

    if message is None:
        raise ValueError("Message is None.")
    await message.reply_text("=== PROCESSING... ===")
    namespace = f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    # tmp_directory = f"/file/{namespace}"
    chunker = FixedGroupChunker(config={"K": 1})
    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=encoder,
        chunker=chunker,
        namespace=namespace,
        overwrite=False,
    )
    rag = OneRAG(
        vector_collection=vm,
        chat_cache=chat_memory[namespace],
        encoder=encoder,
        llm=llm,
        top_n=10,
    )
    # Generate the response
    query_text = message.text.strip()
    query_text = query_text.replace("/ask ", "")
    if context.user_data and "media_markdown" in context.user_data:
        query_text += f"\n\n{context.user_data['media_markdown']}"

    # answer = rag(query=query_text, user_identifier=namespace)
    answer = await rag.invoke(query=query_text, user_identifier=namespace)
    answer = escape_markdown(answer)
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
                current_chunk, is_close = handle_triple_ticks(current_chunk, is_close)
                _ = await message.reply_text(current_chunk, parse_mode=ParseMode.MARKDOWN_V2)
                current_chunk = ""
        # Handle the last chunk.
        if current_chunk:
            current_chunk, is_close = handle_triple_ticks(current_chunk, is_close)
            _ = await message.reply_text(current_chunk, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        _ = await message.reply_text(answer, parse_mode=ParseMode.MARKDOWN_V2)

    identifier = f"{message.chat.id}/{message.message_id}/r{int(time.time())}"
    ts = str(datetime.now())
    metadata = {
        "username": llm.model_name,
        "isAnswer": True,
        "created": ts,
        "lastUpdated": ts,
        "label": "chat",
    }
    add_to_vector_memory(
        namespace,
        identifier,
        answer,
        metadata,
    )


async def summary_handler(update: Update, context: CallbackContext) -> None:
    if context.user_data and "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)
    if message is None:
        raise ValueError("Message is None.")
    await message.reply_text("=== PROCESSING... ===")
    namespace = f"g{message.chat.id}" if message.chat.id < 0 else str(message.chat.id)
    # tmp_directory = f"/file/{namespace}"
    summarizer = local.Text_to_Text(
        connection_string=config.CONNECTION_STRING, **config.summarizer_t2t_config
    )
    chunker = FixedGroupChunker(config={"K": 1})
    vm = ChromaMemory(
        vdb=main_vdb,
        encoder=encoder,
        chunker=chunker,
        namespace=namespace,
        overwrite=False,
    )
    rag = OneRAG(
        vector_collection=vm,
        chat_cache=chat_memory[namespace],
        encoder=encoder,
        llm=summarizer,
        top_n=30,
    )
    query_text: str = message.text.strip()
    query_text = query_text.replace("/summary ", "")

    if query_text:
        query_text = "Topic/Keyword: General Summary"
    else:
        query_text = f"Topic/Keyword: {query_text}"

    # if "media_markdown" in context.user_data:
        # query_text += f"\n\nMedia: {context.user_data['media_markdown']}"

    answer = await rag.invoke(query=query_text, user_identifier=namespace)
    answer = escape_markdown(answer)
    if len(answer) >= 4096:
        sections = re.split(r"\n{2,}", answer)
        current_chunk = ""
        is_close: bool = True
        for section in sections:
            if len(current_chunk) + len(section) + 2 <= 4096:
                current_chunk += section + "\n\n"
            else:
                current_chunk, is_close = handle_triple_ticks(current_chunk, is_close)
                _ = await message.reply_text(current_chunk, parse_mode=ParseMode.MARKDOWN_V2)
                current_chunk = ""
        # Handle the last chunk.
        if current_chunk:
            current_chunk, is_close = handle_triple_ticks(current_chunk, is_close)
            _ = await message.reply_text(current_chunk, parse_mode=ParseMode.MARKDOWN_V2)
    else:
        _ = await message.reply_text(answer, parse_mode=ParseMode.MARKDOWN_V2)
        
    identifier = f"{message.chat.id}/{message.message_id}/r{int(time.time())}"
    ts = str(datetime.now())
    metadata = {
        "username": llm.model_name,
        "isAnswer": True,
        "created": ts,
        "lastUpdated": ts,
        "label": "chat",
    }
    add_to_vector_memory(
        namespace,
        identifier,
        answer,
        metadata,
    )


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
