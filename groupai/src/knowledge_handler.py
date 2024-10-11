import os
import io
import telegram
import tiktoken
import openai
import logging
from typing import List, Optional
from copy import deepcopy

from model import CompactMessage, Media
from scraper.audio_scraper import AudioToMarkdownScraper
from scraper.docx_scraper import DocxToMarkdownScraper
from scraper.pdf_scraper import PDFToMarkdownScraper
from scraper.image_scraper import ImageToMarkdownScraper

logger = logging.getLogger(__name__)


def text_to_embedding(text: str, api_key: str, model: str = "text-embedding-3-small"):
    # TODO: Check Size!
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"text_to_embedding: {e}")
        raise


async def extract_media(message: telegram.Message) -> Media:
    """Extract media information from a Message."""
    media = Media(isMedia=False, fileid=None, filename=None,
                  mime_type=None, markdown=None)
    if message.document:
        media = Media(
            isMedia=True,
            fileid=message.document.file_id,
            filename=message.document.file_name,
            mime_type=message.document.mime_type,
            markdown=None
        )
    elif message.photo:
        media = Media(
            isMedia=True,
            fileid=message.photo[-1].file_id,
            filename=message.caption,
            mime_type=None,
            markdown=None
        )
    elif message.video:
        media = Media(
            isMedia=True,
            fileid=message.video.file_id,
            filename=message.video.file_name,
            mime_type=message.video.mime_type,
            markdown=None
        )
    elif message.audio:
        media = Media(
            isMedia=True,
            fileid=message.audio.file_id,
            filename=message.audio.file_name,
            mime_type=message.audio.mime_type,
            markdown=None
        )
    elif message.voice:
        media = Media(
            isMedia=True,
            fileid=message.voice.file_id,
            filename=message.voice.file_unique_id,
            mime_type=message.voice.mime_type,
            markdown=None
        )
    return media


async def media_to_transcript(input_file, identifier: str, caption: str, mime_type: str, **kwargs) -> str:
    """Convert media to transcript."""
    ext = mime_type.split('/')[1]
    tmp_file = None
    output = f'❌: Unsupported File Type'
    if "audio/" in mime_type:
        openai_api_key = kwargs.get('OPENAI_API_KEY')
        model = kwargs.get('audio_model', 'whisper-1')
        buffer = io.BytesIO()
        await input_file.download_to_memory(buffer)
        scraper = AudioToMarkdownScraper(
            OPENAI_API_KEY=openai_api_key, model=model)
        md_content = scraper(buffer, identifier, mime_type)
        output = f'🎤: *{md_content}*'
    elif "image/" in mime_type:
        openai_api_key = kwargs.get('OPENAI_API_KEY')
        model = kwargs.get('vision_model', 'gpt-4o-mini')
        tmp_path = f"/file/{identifier.replace('/', '_')}.{ext}"
        await input_file.download_to_drive(tmp_path)
        scraper = ImageToMarkdownScraper(
            OPENAI_API_KEY=openai_api_key, model=model)
        md_content = scraper(tmp_path, caption)
        output = f'🖼: *{md_content}*'
    elif mime_type == "application/pdf":
        tmp_path = f"/file/{identifier.replace('/', '_')}.pdf"
        await input_file.download_to_drive(tmp_path)
        scraper = PDFToMarkdownScraper()
        md_content = scraper(tmp_path)
        output = f'📑: *{md_content}*'
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        tmp_path = f"/file/{identifier.replace('/', '_')}.docx"
        await input_file.download_to_drive(tmp_path)
        scraper = DocxToMarkdownScraper()
        md_content = scraper(tmp_path)
        output = f'📄: *{md_content}*'
    if tmp_file:
        os.remove(tmp_file)
    return output


async def parse_message(message: telegram.Message, edited: bool = False) -> CompactMessage:
    """
    Parse a Message object from Telegram API into a CompactMessage object.

    Args:
    - message (Message): The Message object to parse.
    - edited (bool, optional): If the message is edited. Defaults to False.

    Returns:
    - CompactMessage: The parsed CompactMessage object.

    Notes:
    Fields `isForwarded`, `author` and `isBot` are only applicable when it's a forwarded message.
    """
    msg = CompactMessage(
        identifier=f"{message.chat.id}/{message.message_id}",
        text=message.text if message.text else message.caption if message.caption else "BLANK",
        chattype=message.chat.type,
        chatid=message.chat.id,
        chatname=message.chat.title,
        userid=message.from_user.id,
        username=message.from_user.username or f"{message.from_user.first_name} {message.from_user.last_name}",
        message_id=message.message_id,
        created=str(message.date),
        lastUpdated=str(message.date),
        edited=edited,
        isForwarded=False,
        media=await extract_media(message),
    )
    # Handle Forwarded Message
    forward_origin = getattr(message, "forward_origin", None)
    if forward_origin:
        msg.isForwarded = True
        if forward_origin.type is telegram.constants.MessageOriginType.HIDDEN_USER:
            msg.author = forward_origin.sender_user_name
        else:
            msg.author = forward_origin.sender_user.username or f"{forward_origin.sender_user.first_name} {forward_origin.sender_user.last_name}"
            msg.isBot = forward_origin.sender_user.is_bot

    return msg


class KnowledgeHandler:
    def __init__(
        self, openai_api_key: str, 
        embedding_model: str = "text-embedding-3-small", embedding_chunk_size: int = 4096, stride_rate: float = 0.7,
        gpt_model: str = "gpt-4o-mini", context_window: int = 128000,
        vision_model: str = "gpt-4o-mini", 
        audio_model: str = "whisper-1",
    ):
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.embedding_chunk_size = embedding_chunk_size
        self.stride = int(embedding_chunk_size * stride_rate)
        self.gpt_model = gpt_model # This is not used in current implementation, will be needed when we decided to summary long passages
        self.context_window = context_window
        self.vision_model = vision_model
        self.audio_model = audio_model
    
    async def process_tlg_message(self, message: telegram.Message, is_edited: bool = False, context: telegram.ext.CallbackContext = None):
        processed: CompactMessage = await parse_message(message, is_edited)
        processed_str = str(processed)
        if processed.media.isMedia:
            media_file = await context.bot.get_file(processed.media.fileid)
            media_content_markdown = await self.convert_media_to_markdown(media_file, processed.media.mime_type, processed.identifier, processed_str)
            processed.media.markdown = media_content_markdown
        return processed
    
    def split_media_to_documents(self, base: CompactMessage, media_content_markdown: str):
        tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        token_count = len(tokenizer.encode(media_content_markdown))
        chunks = self.text_to_chunks(media_content_markdown) if token_count > self.embedding_chunk_size else [media_content_markdown]
        documents = self.rebundle_media_chunks(base, chunks)
        return documents
            
    def prepare_documents_for_rag_indexing(self, documents: List[CompactMessage]):
        """
        Notes:
        - As this is used for chatbot where there's a semantic meaning of the message, chunking it is not ideal.
        - It will only be chunked if the message is too long which is quite impossible due to the input limit of a single Telegram message.
        """
        tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        ids = []
        metadatas = []
        chunks = []
        embeddings = []
        for document in documents:
            document_str = str(document)
            token_count = len(tokenizer.encode(document_str))
            text_chunks = self.text_to_chunks(document_str) if token_count > self.embedding_chunk_size else [document_str]
            for index, i_chunk in text_chunks:
                # Generate a text embedding for the message content
                i_embedding = self.convert_text_to_embedding(i_chunk)
                identifier = f'{document.identifier}|{index}'
                metadata = {
                    "message_id": document.message_id,
                    "sender": document.username, "is_edited": document.edited, "is_deleted": document.deleted,
                    "is_bot": document.isBot, "is_forwarded": document.isForwarded, "author": document.author,
                    "is_media": document.media.isMedia, "mime_type": document.media.mime_type,
                    "is_answer": document.isAnswer, "created": document.created, "lastUpdated": document.lastUpdated
                }
                ids.append(identifier)
                metadatas.append(metadata)
                embeddings.append(i_embedding)
                chunks.append(i_chunk)
            return ids, metadatas, chunks, embeddings

    async def convert_media_to_markdown(self, input_file, mime_type: str, identifier: str, caption: str):
        return await media_to_transcript(
            input_file, identifier, caption, mime_type, OPENAI_API_KEY=self.openai_api_key, 
            vision_model=self.vision_model, audio_model=self.audio_model
        )
        
    async def convert_text_to_embedding(self, text):
        return text_to_embedding(text, self.openai_api_key, model=self.embedding_model)
    
    def text_to_chunks(self, text: str):
        chunks: List[str] = []
        for start in range(0, len(text), self.embedding_chunk_size * self.stride):
            end = min(start + self.embedding_chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
        return chunks
    
    def rebundle_media_chunks(self, base: CompactMessage, chunks: List[str]):
        messages: List[CompactMessage] = []
        for i, chunk in enumerate(chunks):
            copy_message = deepcopy(base)
            copy_message.media.markdown = chunk
            copy_message.identifier = f'{base.identifier}|{i}'
            messages.append(copy_message)
        return messages
    
        
    async def __call__(self, message: telegram.Message, context: telegram.ext.CallbackContext, edited: bool = False):
        processed_message = await self.process_tlg_message(message, edited, context)
        if processed_message.media.isMedia:
            documents = self.split_media_to_documents(processed_message, processed_message.media.markdown)
        else:
            documents = [processed_message]
        ids, metadatas, chunks, embeddings = self.prepare_documents_for_rag_indexing(documents)
        return ids, metadatas, chunks, embeddings
