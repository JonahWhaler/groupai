# __all__ = ["audio_scraper", "docx_scraper", "pdf_scraper"]

import os
import io
from typing import Optional
from audio_scraper import AudioToMarkdownScraper
from docx_scraper import DocxToMarkdownScraper
from pdf_scraper import PDFToMarkdownScraper
from image_scraper import ImageToMarkdownScraper


def get_scraper(mime_type: str, **kwargs):
    if "audio/" in mime_type:
        return AudioToMarkdownScraper(mime_type=mime_type, **kwargs)
    if "image/" in mime_type:
        return ImageToMarkdownScraper(**kwargs)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return DocxToMarkdownScraper(
            image_interpretor=ImageToMarkdownScraper(**kwargs), **kwargs
        )
    if mime_type == "application/pdf":
        return PDFToMarkdownScraper(
            image_interpretor=ImageToMarkdownScraper(**kwargs), **kwargs
        )
    return None


async def media_to_transcript(
    input_file: str, filename_wo_ext: str, caption: str, mime_type: str,
    tmp_directory: Optional[str] = None,
    **kwargs
) -> str:
    """Convert media to transcript."""
    scraper = get_scraper(mime_type, tmp_directory=tmp_directory, **kwargs)
    assert scraper is not None

    ext = mime_type.split('/')[1]
    tmp_file = None
    md_content = f'<error>\nUnsupported File Type\n</error>'
    buffer = None

    # tmp_path = f"{tmp_directory}/{filename_wo_ext}.{ext}"
    # await input_file.download_to_drive(tmp_path)
    md_content = scraper(
        input_path=input_file, buffer=buffer,
        identifier=filename_wo_ext, caption=caption, output_path="/mnt/second/projects/groupai/groupai/src/scraper/output.md"
    )

    # if tmp_file:
    #     os.remove(tmp_file)
    return md_content


async def test(OPENAI_API_KEY, vision_model, audio_model, file_path, mime_type, caption):
    filename_wo_ext: str = os.path.basename(file_path).split(".")[0]
    md = await media_to_transcript(
        input_file=file_path, filename_wo_ext=filename_wo_ext,
        caption=caption, mime_type=mime_type,
        OPENAI_API_KEY=OPENAI_API_KEY,
        vision_model=vision_model, audio_model=audio_model,
        tmp_directory="/mnt/second/projects/groupai/groupai/src/scraper"
    )
    print(md)


if __name__ == "__main__":
    import asyncio

    OPENAI_API_KEY = ""
    vision_model = "gpt-4o-mini-2024-07-18"
    audio_model = "whisper-1"
    file_path = ""
    mime_type = ""
    caption = ""
    asyncio.run(test(OPENAI_API_KEY, vision_model,
                audio_model, file_path, mime_type, caption))
