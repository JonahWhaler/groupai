import fitz  # PyMuPDF
import pdfplumber
import openai
import os
import io
from typing import Optional, List, Dict
from .base_scraper import BaseScraper
from .image_scraper import ImageToMarkdownScraper
# import pandas as pd
# import warnings
# # Suppress pandas' FutureWarning
# warnings.simplefilter(action='ignore', category=FutureWarning)


class PDFToMarkdownScraper(BaseScraper):
    """
    Scrape the pdf file and convert information it contains to markdown.

    Coverage
    ------
    - text content
    - tables
    - links
    - images
    """

    def __init__(self, **kwargs):
        self.image_interpretor: Optional[ImageToMarkdownScraper] = kwargs.get(
            "image_interpretor", None)
        self.tmp_directory = kwargs.get("tmp_directory", None)
        os.makedirs(self.tmp_directory, exist_ok=True)

    def __call__(
        self, input_path: Optional[str] = None, buffer: Optional[io.BytesIO] = None,
        identifier: Optional[str] = None, caption: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        assert input_path is not None and input_path != ""
        assert input_path[-4:] == ".pdf"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        output_markdown = self.to_markdown(input_path)
        if output_path is not None:
            self.save_markdown(output_markdown, output_path)
        return output_markdown

    def extract_alt_text(self, page, img):
        """Attempt to extract alt text for an image."""
        alt_text = None

        # Try to get alt text from the image object
        if 'alt' in img:
            alt_text = img['alt']

        # If not found, try to get it from the page's /MC (marked content) entries
        if alt_text is None:
            for xref in page.get_contents():
                stream = page.parent.xref_stream(xref)
                if stream:
                    # Simplified check for alt text in marked content
                    if b'/Alt' in stream:
                        alt_parts = stream.split(b'/Alt')
                        if len(alt_parts) > 1:
                            alt_text = alt_parts[1].split(b')')[0].decode(
                                'utf-8', errors='ignore').strip()
                            break

        return alt_text if alt_text else "No alt text found"

    def extract_image_description(self, image_bytes, image_name, alt_text):
        if self.image_interpretor is None:
            return "Image description not available"
        image_stream = io.BytesIO(image_bytes)
        # Store locally
        image_stream.seek(0)
        tmp_path = f"{self.tmp_directory}/{image_name}"
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)
        image_caption = f"alt_text={alt_text},filename={image_name}. This is an attachment found in a pdf file."
        image_description = self.image_interpretor.to_markdown(
            image_path=tmp_path, caption=image_caption
        )
        # Remove locally
        os.remove(tmp_path)
        return image_description
    
    def to_markdown(self, pdf_path: str):

        markdown_content = []

        # Extract text, links, and images using PyMuPDF
        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text()
                links = page.get_links()
                images = page.get_images()

                markdown_content.append(f"# Page {page_number}\n")
                markdown_content.append(text)

                # Extract links if available
                if links:
                    markdown_content.append("\n## Links\n")
                    for link in links:
                        if 'uri' in link:
                            markdown_content.append(
                                f"- [Link on Page {page_number}]({link['uri']})\n")

                # Extract images and their alt text if available
                if images:
                    markdown_content.append("\n## Images\n")
                    for img_index, img in enumerate(images, start=1):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            image_ext = base_image["ext"]
                            image_name = f"image_{page_number}_{img_index}.{image_ext}"
                            alt_text = self.extract_alt_text(page, img)
                            image_description = self.extract_image_description(base_image['image'], image_name, alt_text)
                            markdown_content.append(
                                f"- Image {img_index} on Page {page_number}: {image_name}\n")
                            markdown_content.append(
                                f"  Alt Text: {alt_text}\n")
                            markdown_content.append(
                                f"  Description: \n{image_description}\n")
                            markdown_content.append(
                                f"  [IMAGE ATTACHED: {image_name}]\n")

        # Extract tables using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_index, table in enumerate(tables, start=1):
                    if table:
                        markdown_content.append(
                            f"\n## Table {table_index} on Page {page_number}\n")
                        # markdown_content.append(
                        #     "| " + " | ".join(table[0]) + " |")
                        # markdown_content.append(
                        #     "|" + "|".join(["---"] * len(table[0])) + "|")
                        for row in table[:]:
                            markdown_content.append(
                                "| " + " | ".join(str(cell) for cell in row) + " |")
                        markdown_content.append("\n")

        return '\n'.join(markdown_content)


# Usage example
if __name__ == "__main__":
    OPENAI_API_KEY = ""
    model = "gpt-4o-mini-2024-07-18"
    image_scraper = ImageToMarkdownScraper(OPENAI_API_KEY=OPENAI_API_KEY, vision_model=model)
    
    pdf_scraper = PDFToMarkdownScraper(image_interpretor=image_scraper, tmp_directory="./asset")
    markdown_output = pdf_scraper("./asset/chart-table.pdf")
    # print(markdown_output)
    pdf_scraper.save_markdown(markdown_output, "./asset/chart-table.md")
