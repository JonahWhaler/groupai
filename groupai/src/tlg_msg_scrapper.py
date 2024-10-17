import os
import io
import logging
import telegram
from typing import Any, Optional, List, Tuple

from model import CompactMessage, Media
from scraper import media_to_transcript

logger = logging.getLogger(__name__)


class TlgMsgScraper:
    def __init__(self, tmp_directory: str, **kwargs):
        os.makedirs(tmp_directory, exist_ok=True)
        self.embedding_model = kwargs.get(
            "embedding_model", "text-embedding-3-small")
        self.gpt_model = kwargs.get("gpt_model", "gpt-4o-mini")
        self.audio_model = kwargs.get("audio_model", "whisper-1")
        self.vision_model = kwargs.get("vision_model", "gpt-4o-mini")
        self.tmp_directory = tmp_directory

    async def parse_media(self, message: telegram.Message) -> Media:
        media = Media(
            isMedia=False, fileid=None, filename=None, mime_type=None, markdown=None
        )
        if message.document:
            media = Media(
                isMedia=True,
                fileid=message.document.file_id,
                filename=message.document.file_name,
                mime_type=message.document.mime_type,
                markdown=None
            )
        elif message.photo:
            filename = f"{message.photo[-1].file_id}.jpg"
            media = Media(
                isMedia=True,
                fileid=message.photo[-1].file_id,
                filename=filename,
                mime_type="image/jpeg",
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

    async def parse_message(
        self, message: telegram.Message, edited: bool = False
    ) -> CompactMessage:
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
            media=await self.parse_media(message),
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

    async def preprocessing(self, message: telegram.Message, edited: bool = False) -> CompactMessage:
        processed: CompactMessage = await self.parse_message(message, edited)
        return processed

    def to_markdown(self, message: CompactMessage) -> str:
        tmp_path = f"{self.tmp_directory}/{message.media.filename}"
        if message.media.isMedia:
            media_markdown = media_to_transcript(
                input_file=tmp_path,
                filename_wo_ext=os.path.basename(message.media.filename),
                caption=message.text,
                mime_type=message.media.mime_type,
                tmp_directory=self.tmp_directory,
                vision_model=self.vision_model,
                audio_model=self.audio_model,
                gpt_model=self.gpt_model
            )
            message.media.markdown = media_markdown
        return str(message)

    async def __call__(self, message: telegram.Message, context: telegram.ext.CallbackContext, edited: bool = False):
        tmp_path = None
        my_msg = await self.preprocessing(message, edited)
        if my_msg.media.isMedia:
            media_file = await context.bot.get_file(my_msg.media.fileid)
            tmp_path = f'{self.tmp_directory}/{my_msg.media.filename}'
            await media_file.download_to_drive(tmp_path)
        markdown = self.to_markdown(my_msg)
        if tmp_path:
            os.remove(tmp_path)
        return markdown
