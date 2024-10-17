from typing import Optional
import io
import os
import math
from pydub import AudioSegment
import openai
import ffmpeg
from .base_scraper import BaseScraper


class AudioToMarkdownScraper(BaseScraper):
    def __init__(self, **kwargs):
        self.model = kwargs.get("audio_model", "whisper-1")
        self.mime_type = kwargs.get("mime_type", None)
        assert self.mime_type is not None
        self.tmp_directory = kwargs.get("tmp_directory", None)
        os.makedirs(self.tmp_directory, exist_ok=True)
        self.gpt_model = kwargs.get("gpt_model", "gpt-4o-mini")

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

    def to_chunks(self, input_file: str, max_size_mb: int = 20, audio_bitrate: str = '128k'):
        slices = []
        try:
            # Get input file information
            probe = ffmpeg.probe(input_file)
            duration = float(probe['format']['duration'])
            
            # Convert audio_bitrate to bits per second
            bitrate_bps = int(audio_bitrate[:-1]) * 1024  # Convert 'xxxk' to bits/second
            
            # Calculate expected output size in bytes
            expected_size_bytes = (bitrate_bps * duration) / 8
            
            # Calculate the number of slices based on expected output size
            num_slices = math.ceil(expected_size_bytes / (max_size_mb * 1024 * 1024))
            
            # Calculate the duration of each slice
            slice_duration = duration / num_slices

            # Convert and slice the audio
            for i in range(num_slices):
                start_time = i * slice_duration
                output_file = os.path.join(self.tmp_directory, f'slice{i+1}.ogg')
                try:
                    # Convert and slice
                    stream = ffmpeg.input(input_file, ss=start_time, t=slice_duration)
                    stream = ffmpeg.output(stream, output_file, acodec='libvorbis', audio_bitrate=audio_bitrate)
                    ffmpeg.run(stream, overwrite_output=True)

                    # Print information about the exported file
                    output_probe = ffmpeg.probe(output_file)
                    output_size = int(output_probe['format']['size']) / (1024 * 1024)  # Size in MB
                    print(f"Exported {output_file}")
                    print(f"Size: {output_size:.2f} MB")

                    # Print progress
                    progress = (i + 1) / num_slices * 100
                    print(f"Progress: {progress:.2f}%")

                    slices.append(output_file)
                except ffmpeg.Error as e:
                    print(f"Error processing slice {i+1}:")
                    if e.stderr is not None:
                        print(e.stderr.decode())
                    else:
                        print(str(e))
            return slices
        except ffmpeg.Error as e:
            print("Error during file processing:")
            if e.stderr is not None:
                print(e.stderr.decode())
            else:
                print(str(e))

    def to_markdown(self, audio_path: str, context: str = "This is the first part of the audio."):
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        buffer = io.BytesIO(audio_data)
        buffer.name = "audio.ogg"
        buffer.seek(0)
        transcribe_response = openai.audio.transcriptions.create(
            file=buffer, model=self.model, response_format='text',
            prompt=f"{context}"
        )
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        cc_response = client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You will be given a piece of transcript together with the summary of the previous part. You will summarize it."},
                {"role": "user", "content": f"<summary>{context}</summary>"},
                {"role": "user", "content": f"<transcript>{transcribe_response}</transcript>"}
            ],
            max_tokens=200,
            temperature=1,
            n=1
        )
        summary = cc_response.choices[0].message.content.strip()
        return transcribe_response, summary

    def temporary_store_media(self, buffer: io.BytesIO, buffer_name: str):
        output_path = f"{self.tmp_directory}/{buffer_name}.ogg"
        buffer.seek(0)
        
        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())
        return output_path

    def __call__(
        self, input_path: Optional[str] = None, buffer: Optional[io.BytesIO] = None,
        identifier: Optional[str] = None, caption: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        assert input_path is not None and input_path != ""
        assert identifier is not None
        with open(input_path, 'rb') as f:
            audio_data = f.read()
        buffer = io.BytesIO(audio_data)
        # Convert to ogg
        ogg_buffer = self.convert_to_ogg_if_necessary(buffer, identifier, self.mime_type)
        
        tmp_path = self.temporary_store_media(buffer=ogg_buffer, buffer_name=str(identifier))
        chunks = self.to_chunks(tmp_path)
        print("\n".join(chunks))
        markdown_chunks = []
        context = "This is the first part of the audio."
        for chunk in chunks:
            markdown_chunk, context = self.to_markdown(chunk, context)
            markdown_chunks.append(markdown_chunk)
        
        # markdown_chunks = list(map(lambda chunk: self.to_markdown(chunk), chunks))
        markdown = ''.join(markdown_chunks)
        # markdown = ""
        # for chunk in chunks:
        #     markdown += self.to_markdown(chunk)
        # Clean Up
        os.remove(tmp_path)
        for chunk in chunks:
            os.remove(chunk)
        if output_path is not None:
            self.save_markdown(markdown, output_path)
        return markdown
