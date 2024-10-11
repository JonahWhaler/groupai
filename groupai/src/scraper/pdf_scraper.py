import fitz  # PyMuPDF
import pdfplumber
# import pandas as pd
import os
from typing import Optional, List, Dict
import warnings

# Suppress pandas' FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


class PDFToMarkdownScraper:
    def __init__(self):
        self.images_found: List[Dict] = []

    def __call__(self, pdf_path, output_path: Optional[str] = None) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        output_markdown = self.extract_to_markdown(pdf_path)
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
                            alt_text = alt_parts[1].split(b')')[0].decode('utf-8', errors='ignore').strip()
                            break
        
        return alt_text if alt_text else "No alt text found"

    def extract_to_markdown(self, pdf_path: str):
        markdown_content = []
        self.images_found = []  # Reset images found for this extraction

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
                            markdown_content.append(f"- [Link on Page {page_number}]({link['uri']})\n")

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

                            self.images_found.append({
                                "page": page_number,
                                "name": image_name,
                                "alt_text": alt_text
                            })

                            markdown_content.append(f"- Image {img_index} on Page {page_number}: {image_name}\n")
                            markdown_content.append(f"  Alt Text: {alt_text}\n")
                            markdown_content.append(f"  [IMAGE ATTACHED: {image_name}]\n")

        # Extract tables using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_index, table in enumerate(tables, start=1):
                    if table:
                        markdown_content.append(f"\n## Table {table_index} on Page {page_number}\n")
                        markdown_content.append("| " + " | ".join(table[0]) + " |")
                        markdown_content.append("|" + "|".join(["---"] * len(table[0])) + "|")
                        for row in table[1:]:
                            markdown_content.append("| " + " | ".join(str(cell) for cell in row) + " |")
                        markdown_content.append("\n")

        return '\n'.join(markdown_content)

    def save_markdown(self, output_markdown: str, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(output_markdown)
        print(f"Markdown content has been extracted and saved to '{output_path}'")
        
        # Print information about attached images
        if self.images_found:
            print("\nAttached Images:")
            for img in self.images_found:
                print(f"- Page {img['page']}: {img['name']} (Alt text: {img['alt_text']})")
        else:
            print("\nNo images found in the PDF.")

# Usage example
if __name__ == "__main__":
    pdf_scraper = PDFToMarkdownScraper()
    markdown_output = pdf_scraper("sample.pdf", "output.md")
    print(markdown_output)
