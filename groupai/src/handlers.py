import logging
import os
import io
from time import time
import telegram
from telegram import Message, Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from typing import Optional, List
import openai
import chromadb
# import tiktoken
# from copy import deepcopy

from storage import SQLite3_Storage
from model import CompactMessage, Media
# import myfunction
from rag.base import BaseRAG
from knowledge_handler import KnowledgeHandler
from tlg_msg_scrapper import TlgMsgScraper

logger = logging.getLogger(__name__)
master = os.getenv("MASTER_TLG_ID", 0)
assert master != 0

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "whisper-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Note!!!
# Conversavative Estimation: 1 token = 1 character.
CONTEXT_WINDOW = 128000
CONTEXT_BUFFER = 1000
CHUNK_SIZE = 4096
MAX_CHUNK_SIZE = 8100

openai.api_key = os.environ["OPENAI_API_KEY"]
vdb_client = chromadb.Client(
    settings=chromadb.Settings(
        is_persistent=True,
        persist_directory="/file",
    )
)


def get_metadata(message: CompactMessage) -> dict:
    metadata = dict()
    metadata["created"] = message.created
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
    logger.info(f"\nMiddleware Function => Update: {update}")
    # Extract the message or edited message from the update
    message: Optional[telegram.Message] = getattr(update, "message", None)
    edited_message: Optional[telegram.Message] = getattr(
        update, "edited_message", None
    )
    if not message and not edited_message:
        logger.error(
            f"\nException: [Message Body Not Found]=> Update: {update}")
        return None
    
    if edited_message:
        context.user_data["edited_message"] = True
        namespace = f'g{edited_message.chat.id}' if edited_message.chat.id < 0 else str(edited_message.chat.id)
    else:
        namespace = f'g{message.chat.id}' if message.chat.id < 0 else str(message.chat.id)
    
    tmp_directory = f'/file/{namespace}'
    tlg_msg_scraper = TlgMsgScraper(
        embedding_model=EMBEDDING_MODEL, embedding_chunk_size=CHUNK_SIZE, stride_rate=0.75,
        gpt_model=GPT_MODEL, context_window=CONTEXT_WINDOW,
        vision_model=VISION_MODEL, audio_model=AUDIO_MODEL, tmp_directory=tmp_directory
    )
    if edited_message:
        processed_message = await tlg_msg_scraper.preprocessing(edited_message, True)
    else:
        processed_message = await tlg_msg_scraper.preprocessing(message, False)

    if processed_message.media.isMedia:
        media_file = await context.bot.get_file(processed_message.media.fileid)
        # Issue: More than one user upload file with the same filename
        tmp_path = f'{tlg_msg_scraper.tmp_directory}/{processed_message.media.filename}'
        await media_file.download_to_drive(tmp_path)
        processed_message.media.markdown = tlg_msg_scraper.to_markdown(
            processed_message)
        os.remove(tmp_path)
        context.user_data['media_markdown'] = processed_message.media.markdown

    # Store the CompactMessage in an SQLite database
    storage = SQLite3_Storage(f"/file/{namespace}.db", overwrite=False)
    if processed_message.edited:
        old_message = storage.get(processed_message.identifier)
        if old_message:
            processed_message.created = old_message['created']
    storage.set(processed_message.identifier, processed_message.to_dict())

    knowledge_handler = KnowledgeHandler(
        tmp_directory=tmp_directory,
        vdb=vdb_client,
        embedding_model=EMBEDDING_MODEL, embedding_chunk_size=CHUNK_SIZE, stride_rate=0.75,
        gpt_model=GPT_MODEL, context_window=CONTEXT_WINDOW
    )
    metadata = get_metadata(processed_message)
    logger.info(f"Metadat: {metadata}")
    logger.info(f"Information: {str(processed_message)}")
    if "edited_message" in context.user_data:
        knowledge_handler.update(namespace=namespace, identifier=processed_message.identifier,
                          knowledge=str(processed_message), metadata=metadata)
    else:
        knowledge_handler.add(namespace=namespace, identifier=processed_message.identifier,
                          knowledge=str(processed_message), metadata=metadata)


async def error_handler(update: object, context: CallbackContext):
    logger.error(msg="Exception while handling an update:",
                 exc_info=context.error)
    logger.info(f"\nError Handler => Update: {update}")


async def help_handler(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("https://github.com/JonahTzuChi/groupai")


async def message_handler(update: Update, context: CallbackContext) -> None:
    if "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)
    await message.reply_text("=== COPY ===")


def escape_markdown_v2(text):
    """
    Escape special characters for Telegram's MarkdownV2.
    """
    # Characters that need to be escaped
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    # Escape backslash first to avoid double escaping
    text = text.replace('\\', '\\\\')
    
    # Escape special characters
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return f"```\n{text}\n```"


async def ask_handler(update: Update, context: CallbackContext) -> None:
    if "edited_message" in context.user_data:
        message = getattr(update, "edited_message", None)
    else:
        message = getattr(update, "message", None)
        
    await message.reply_text("=== PROCESSING... ===")
    namespace = f'g{message.chat.id}' if message.chat.id < 0 else str(message.chat.id)
    tmp_directory = f'/file/{namespace}'
    vector_collection = vdb_client.get_or_create_collection(name=namespace)
    # Instantiate the RAG model
    rag = BaseRAG(
        vector_collection=vector_collection, embedding_model=EMBEDDING_MODEL, gpt_model=GPT_MODEL, top_n=10
    )
    # Generate the response
    query_text = message.text
    if "media_markdown" in context.user_data:
        query_text += f"\n\n{context.user_data['media_markdown']}"

    answer = rag(query_text)
    answer = escape_markdown_v2(answer)
    # Send the response
    logger.info(f"Answer: {answer}")
    reply_msg = await message.reply_text(answer, parse_mode=ParseMode.MARKDOWN_V2)
    tlg_msg_scraper = TlgMsgScraper(
        embedding_model=EMBEDDING_MODEL, embedding_chunk_size=CHUNK_SIZE, stride_rate=0.75,
        gpt_model=GPT_MODEL, context_window=CONTEXT_WINDOW,
        vision_model=VISION_MODEL, audio_model=AUDIO_MODEL, tmp_directory=tmp_directory
    )
    processed_message = await tlg_msg_scraper.preprocessing(reply_msg, False)
    processed_message.isAnswer = True
    # Store the response in an SQLite database
    storage = SQLite3_Storage(f"/file/{namespace}.db", overwrite=False)
    storage.set(processed_message.identifier, processed_message.to_dict())
    # Store the knowledge in the knowledge base
    knowledge_handler = KnowledgeHandler(
        tmp_directory=tmp_directory,
        vdb=vdb_client,
        embedding_model=EMBEDDING_MODEL, embedding_chunk_size=CHUNK_SIZE, stride_rate=0.75,
        gpt_model=GPT_MODEL, context_window=CONTEXT_WINDOW
    )
    metadata = get_metadata(processed_message)
    knowledge_handler.add(namespace=namespace, identifier=processed_message.identifier,
                          knowledge=str(processed_message), metadata=metadata)


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
