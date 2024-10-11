from typing import Optional
import io
from pydub import AudioSegment
import openai


class AudioToMarkdownScraper:
    def __init__(self, OPENAI_API_KEY: str, model: str):
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.model = model
    
    def convert_to_ogg_if_necessary(self, buffer: io.BytesIO, buffer_name: str, mimetype: str) -> io.BytesIO:
        ext = mimetype.split('/')[1]
        
        if ext not in ["ogg", "oga"]:
            audio = AudioSegment.from_file(buffer)
            ogg_stream = io.BytesIO()
            audio.export(ogg_stream, format="ogg")
            
            ogg_stream.seek(0)
            buffer = ogg_stream.getvalue()
            buffer = io.BytesIO(buffer)
        
        buffer.name = f"{buffer_name}.ogg"
        buffer.seek(0)
        return buffer
    
    def convert_to_markdown(self, buffer: io.BytesIO):
        client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        transcribe_response = client.audio.transcriptions.create(
            file=buffer, model=self.model, response_format='text'
        )
        return transcribe_response
    
    def save_markdown(self, output_markdown: str,output_path: str):
        with open(output_path, 'w') as md_file:
            md_file.write(output_markdown)
        print(f"Markdown content has been extracted and saved to '{output_path}'")
    
    def __call__(self, buffer: io.BytesIO, buffer_name: str, mimetype: str) -> str:
        ogg_buffer = self.convert_to_ogg_if_necessary(buffer, buffer_name, mimetype)
        output_markdown = self.convert_to_markdown(ogg_buffer)
        return output_markdown
