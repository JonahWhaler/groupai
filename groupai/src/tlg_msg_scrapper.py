import os

# import io
import logging

# from typing import Any, Optional, List, Tuple

import telegram
from telegram.ext import CallbackContext

from model import CompactMessage, Media  # type: ignore
from llm_agent_toolkit import ImageGenerator, Transcriber, loader

logger = logging.getLogger(__name__)


class TlgMsgScraper:
    """Transform telegram.Message -> Markdown"""

    def __init__(
        self,
        tmp_directory: str,
        image_interpreter: ImageGenerator,
        transcriber: Transcriber,
        **kwargs,
    ):
        os.makedirs(tmp_directory, exist_ok=True)
        self.image_interpreter = image_interpreter
        self.transcriber = transcriber
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
                markdown=None,
            )
        elif message.photo:
            filename = f"{message.photo[-1].file_id}.jpg"
            media = Media(
                isMedia=True,
                fileid=message.photo[-1].file_id,
                filename=filename,
                mime_type="image/jpeg",
                markdown=None,
            )
        elif message.video:
            media = Media(
                isMedia=True,
                fileid=message.video.file_id,
                filename=message.video.file_name,
                mime_type=message.video.mime_type,
                markdown=None,
            )
        elif message.audio:
            media = Media(
                isMedia=True,
                fileid=message.audio.file_id,
                filename=message.audio.file_name,
                mime_type=message.audio.mime_type,
                markdown=None,
            )
        elif message.voice:
            media = Media(
                isMedia=True,
                fileid=message.voice.file_id,
                filename=message.voice.file_unique_id,
                mime_type=message.voice.mime_type,
                markdown=None,
            )
        return media

    async def parse_message(
        self, message: telegram.Message, edited: bool = False
    ) -> CompactMessage:
        user = message.from_user
        if not user:
            raise ValueError(f"Expect User Message, but got {message}.")

        msg = CompactMessage(
            identifier=f"{message.chat.id}/{message.message_id}",
            text=(
                message.text
                if message.text
                else message.caption if message.caption else "BLANK"
            ),
            chattype=message.chat.type,
            chatid=message.chat.id,
            chatname=message.chat.title,
            userid=user.id,
            username=user.username or f"{user.first_name} {user.last_name}",
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
                msg.author = (
                    forward_origin.sender_user.username
                    or f"{forward_origin.sender_user.first_name} {forward_origin.sender_user.last_name}"
                )
                msg.isBot = forward_origin.sender_user.is_bot

        return msg

    async def preprocessing(
        self, message: telegram.Message, edited: bool = False
    ) -> CompactMessage:
        processed: CompactMessage = await self.parse_message(message, edited)
        return processed

    def to_markdown(
        self, message: CompactMessage, input_path: str | None = None
    ) -> str:
        if message.media.isMedia and input_path:
            config = {
                "text_only": False,
                "tmp_directory": self.tmp_directory,
                "image_interpreter": self.image_interpreter,
            }
            if message.media.mime_type == "application/pdf":
                ldr = loader.PDFLoader(**config)
                content = ldr.load(input_path=input_path)
            elif (
                message.media.mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                ldr = loader.MsWordLoader(**config)
                content = ldr.load(input_path=input_path)
            elif message.media.mime_type.startswith("image/"):
                ldr = loader.ImageToTextLoader(
                    self.image_interpreter,
                    prompt="What's in the image? Describe in details.",
                )
                content = ldr.load(input_path=input_path)
            elif message.media.mime_type.startswith("text/"):
                ldr = loader.TextLoader(encoding="utf-8")
                content = ldr.load(input_path=input_path)
            elif message.media.mime_type.startswith("audio/"):
                if self.transcriber:
                    responses = self.transcriber.transcribe(
                        prompt="What's in the audio file?",
                        filepath=input_path,
                        tmp_directory=self.tmp_directory,
                    )
                    content = ""
                    for response in responses:
                        content += response["content"]
                else:
                    content = "Transcript not available."
            else:
                raise ValueError(f"'{message.media.mime_type}' is not supported.")
            logger.info(
                "Load %s\n%s\nfull len = %d", input_path, content[:100], len(content)
            )
            message.media.markdown = content
        return str(message)

    async def __call__(
        self,
        message: telegram.Message,
        context: CallbackContext,
        edited: bool = False,
    ):
        tmp_path = None
        my_msg = await self.preprocessing(message, edited)
        # if my_msg.media.isMedia:
        #     media_file = await context.bot.get_file(my_msg.media.fileid)
        #     tmp_path = f"{self.tmp_directory}/{my_msg.media.filename}"
        #     await media_file.download_to_drive(tmp_path)
        markdown = self.to_markdown(my_msg, tmp_path)
        if tmp_path:
            os.remove(tmp_path)
        return markdown
