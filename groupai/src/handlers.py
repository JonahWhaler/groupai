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
import tiktoken
from copy import deepcopy

from storage import SQLite3_Storage
from model import CompactMessage, Media
import myfunction
from rag.base import BaseRAG
from knowledge_handler import KnowledgeHandler

logger = logging.getLogger(__name__)
master = os.getenv("MASTER_TLG_ID", 0)
assert master != 0

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Note!!!
# Conversavative Estimation: 1 token = 1 character.
CONTEXT_WINDOW = 128000
CONTEXT_BUFFER = 1000
CHUNK_SIZE = 4096
MAX_CHUNK_SIZE = 8100

openai.api_key = os.getenv("OPENAI_TOKEN")
vdb_client = chromadb.Client(
    settings=chromadb.Settings(
        is_persistent=True,
        persist_directory="/file",
    )
)


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

    Flow:
        1. Log the incoming update.
        2. Extract the message or edited message from the update.
        3. Parse the message into a CompactMessage format using myfunction.parse_message().
           This includes handling forwarded messages and extracting media information.
        4. If media is present:
           - Extract media details using extract_media() function.
           - Transcribe or describe the media content using media_to_transcript() function.
        5. Store the CompactMessage in an SQLite database, keyed by chat ID.
        6. Generate a text embedding for the message content using OpenAI's API.
        7. Store the embedding, along with metadata, in a vector database collection named after the chat ID.

    Media Handling:
        - Supports documents, photos, videos, audio files, and voice messages.
        - Media information is extracted into a Media object with properties:
          isMedia, fileid, filename, mime_type, and transcript.
        - Only the content of parseable media is stored, not the file itself:
          * Audio: Transcribed using OpenAI's Whisper model.
          * Images: Described using GPT-4 Vision model.
          * PDFs: Converted to markdown.
          * DOCX files: Converted to markdown.
        - The file_id is stored, allowing retrieval from Telegram's servers when needed.
        - Transcripts are prefixed with emojis indicating the media type (e.g., 🎤 for audio, 🖼 for images).
        - Temporary files may be created during processing but are deleted afterwards.

    File Storage:
        - The original files are NOT stored locally.
        - Only the file_id and processed content (transcriptions, descriptions) are stored.
        - This approach saves storage space while maintaining the ability to access original files via Telegram's API.

    Notes:
        - Uses myfunction.parse_message() to convert Telegram Message to CompactMessage.
        - Handles both new and edited messages.
        - Special handling for forwarded messages, including original sender information.
        - Media processing includes transcription/description using various models and techniques.
        - SQLite database filename is based on the chat ID.
        - Vector database collection name is the chat ID, prefixed with 'g' for group chats.
        - Logs errors if the message body is not found.
        - Assumes the existence of a configured logger and vdb_client.

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

    knowledge_handler = KnowledgeHandler(
        openai_api_key=openai.api_key, embedding_model=EMBEDDING_MODEL, embedding_chunk_size=CHUNK_SIZE, stride_rate=0.75,
        gpt_model=GPT_MODEL, context_window=CONTEXT_WINDOW,
        vision_model="gpt-4o-mini", audio_model="whisper-1",
    )
    if edited_message:
        processed_message = await knowledge_handler.process_tlg_message(edited_message, True, context)
    else:
        processed_message = await knowledge_handler.process_tlg_message(message, False, context)
    # processed_message is a complete CompactMessage object, media content is not chunked
    
    # Store the CompactMessage in an SQLite database
    storage = SQLite3_Storage(f"/file/{processed_message.chatid}.db", overwrite=False)
    storage.set(processed_message.identifier, processed_message.to_dict())
    
    # Embedding model has size limitation, so we need to split the document into chunks
    if processed_message.media.isMedia:
        documents = knowledge_handler.split_media_to_documents(processed_message, processed_message.media.markdown)
    else:
        documents = [processed_message]
    
    # Identify the vector database collection
    chatid = f'g{processed_message.chatid}' if processed_message.chatid< 0 else str(processed_message.chatid)
    vector_collection = vdb_client.get_or_create_collection(name=chatid,)
    # Store the embedding, along with metadata, in a vector database collection
    ids, metadatas, chunks, embeddings = knowledge_handler.prepare_documents_for_rag_indexing(documents)
    vector_collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)


async def error_handler(update: object, context: CallbackContext):
    logger.error(msg="Exception while handling an update:",
                 exc_info=context.error)
    logger.info(f"\nError Handler => Update: {update}")


async def help_handler(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("https://github.com/JonahTzuChi/groupai")


async def message_handler(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("=== COPY ===")
    # pass


async def ask_handler(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("=== PROCESSING... ===")
    chatid = update.message.chat.id
    if chatid < 0:
        chatid = f'g{chatid}'
    else:
        chatid = str(chatid)
    
    # Identify the vector database collection
    vector_collection = vdb_client.get_or_create_collection(name=chatid)
    # Instantiate the RAG model
    rag = BaseRAG(
        vector_collection=vector_collection, openai_api_key=openai.api_key,
        embedding_model=EMBEDDING_MODEL, gpt_model=GPT_MODEL, top_n=20
    )
    # Generate the response
    query_text = update.message.text
    answer = rag(query_text)
    # Send the response
    reply_msg = await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
    # Store the response in an SQLite database
    conversation = CompactMessage(
        identifier=f"{reply_msg.chat.id}/{reply_msg.message_id}",
        text=None,
        chattype=reply_msg.chat.type,
        chatid=reply_msg.chat.id,
        chatname=reply_msg.chat.title or f"{reply_msg.chat.first_name} {reply_msg.chat.last_name}",
        userid=reply_msg.from_user.id,
        username=reply_msg.from_user.username,
        message_id=reply_msg.message_id,
        created=str(reply_msg.date),
        lastUpdated=str(reply_msg.date),
        edited=False,
        deleted=False,
        isForwarded=False,
        author=None,
        isBot=False,
        isAnswer=True,
        media=Media(False, None, None, None, None)
    )
    storage = SQLite3_Storage(
        f"/file/{conversation.chatid}.db", overwrite=False)
    storage.set(conversation.identifier, conversation.to_dict())
    # Generate a text embedding for the message content
    knowledge_handler = KnowledgeHandler(
        openai_api_key=openai.api_key, embedding_model=EMBEDDING_MODEL, embedding_chunk_size=CHUNK_SIZE, stride_rate=0.75,
        gpt_model=GPT_MODEL, context_window=CONTEXT_WINDOW,
        vision_model="gpt-4o-mini", audio_model="whisper-1",
    )
    ids, metadatas, chunks, embeddings = knowledge_handler.prepare_documents_for_rag_indexing([conversation])
    vector_collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)


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
