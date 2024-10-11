from docx import Document
import re
from io import StringIO
from typing import Optional


class DocxToMarkdownScraper:
    def __init__(self):
        pass

    def get_cell_content_with_formatting(self, cell):
        content = StringIO()
        for para in cell.paragraphs:
            for run in para.runs:
                if run.bold:
                    content.write(f"**{run.text}**")
                elif run.italic:
                    content.write(f"*{run.text}*")
                else:
                    content.write(run.text)
            content.write('\n')
        return content.getvalue().strip()

    def convert_to_markdown(self, doc: Document):
        markdown = []

        # Iterate through all elements in the document
        for para in doc.paragraphs:
            # Handle headings
            if para.style.name.startswith('Heading'):
                level = int(re.search(r'\d+', para.style.name).group(0))
                markdown.append(f"{'#' * level} {para.text}")
            else:
                # Handle regular paragraphs and text
                if para.text.strip():
                    markdown.append(para.text)

        # Extract tables
        for table in doc.tables:
            markdown.append('\n')
            for row_index, row in enumerate(table.rows):
                row_data = [f"| {self.get_cell_content_with_formatting(cell)} " for cell in row.cells]
                markdown.append(''.join(row_data) + '|')
                # Create a separator for header row (assuming first row is header)
                if row_index == 0:
                    header_separator = ''.join(['| --- ' for _ in row.cells])
                    markdown.append(header_separator + '|')
            # Add a newline after the table
            markdown.append('\n')

        # Return markdown as a single joined string
        return '\n'.join(markdown)

    def save_markdown(self, output_markdown: str,output_path: str):
        with open(output_path, 'w') as md_file:
            md_file.write(output_markdown)
        print(f"Markdown content has been extracted and saved to '{output_path}'")

    def __call__(self, docx_path: str, output_path: Optional[str] = None) -> str:
        doc = Document(docx_path)
        output_markdown = self.convert_to_markdown(doc)
        if output_path is not None:
            self.save_markdown(output_markdown, output_path)
        return output_markdown


if __name__  == '__main__':
    # Usage
    converter = DocxToMarkdownScraper()
    converter('example.docx', 'output.md')
