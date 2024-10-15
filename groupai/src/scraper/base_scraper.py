import os
import io
from abc import ABC, abstractmethod
from typing import Optional


class BaseScraper(ABC):
    """
    Scrape the file and convert information it contains to markdown.
    """
    @abstractmethod
    def to_markdown(self):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(
        self, input_path: Optional[str] = None, buffer: Optional[io.BytesIO] = None,
        identifier: Optional[str] = None, caption: Optional[str] = None, 
        output_path: Optional[str] = None
    ) -> str:
        raise NotImplementedError
    
    def save_markdown(self, markdown_input: str, path: str, overwrite: bool = False):
        if overwrite:
            if os.path.exists(path):
                os.remove(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown_input)
        print(f"Markdown content has been extracted and saved to '{path}'")
        