from typing import Optional
import base64
import os
import openai
from base_scraper import BaseScraper
import io


class ImageToMarkdownScraper(BaseScraper):
    # TODO: Ensure the image does not exceed the size limit
    def __init__(self, **kwargs):
        openai.api_key = kwargs.get('OPENAI_API_KEY')
        self.model = kwargs.get("vision_model", "gpt-4o-mini")

    def to_markdown(self, image_path: str, caption: str):
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the message for the API
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the content of this image in 5 to 7 sentences. If familiar faces are present, attempt to identify the faces in the image."
                    }
                ]  
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]

        if caption:
            messages[1]['content'].append({"type": "text", "text": caption})

        # Call the OpenAI API
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300
        )

        # Extract the markdown description
        markdown_description = response.choices[0].message.content
        return markdown_description

    def __call__(
        self, input_path: Optional[str] = None, buffer: Optional[io.BytesIO] = None,
        identifier: Optional[str] = None, caption: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        assert input_path is not None and input_path != ""
        # Validate image file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Image file not found: {input_path}")
        markdown_output = self.to_markdown(input_path, caption)
        if output_path is not None:
            self.save_markdown(markdown_output, output_path)
        return f"<image>\n{markdown_output}\n</image>"


# Usage example
if __name__ == "__main__":
    # OPENAI_API_KEY = ""
    OPENAI_API_KEY = ""
    
    model = "gpt-4o-mini-2024-07-18"
    image_scraper = ImageToMarkdownScraper(OPENAI_API_KEY=OPENAI_API_KEY, vision_model=model)

    # for image in os.listdir("./asset"):
    #     if image.split('.')[1] != 'md':
    #         image_path = os.path.join("./asset", image)
    #         output_path = image_path.replace(".jpg", ".md")
    #         markdown_output = image_scraper(image_path, image, output_path)

    
#     caption = """
# 校長教師愛灑 有趣學習更有效 | 新聞
# https://www.facebook.com/DaAiChannelMalaysia.chn/videos/547851927940870

# 慈濟教育團隊在藍毘尼魯潘德希縣（Rupandehi District）的悉達多市（Bhairahawa）和布德沃爾（Butwal）舉辦私立學校校長和教師愛灑，傳授五i教學法，把學堂變得有趣，學生更有興趣上學去！

# YouTube: https://youtu.be/nHThzKoeC2o
#     """
    image_path = "./asset/waiting-at-airport.JPG"
    caption = "Not Available"

    output_path = "waiting-at-airport.md"
    # print(f"{image_path} => {output_path}")
    # markdown_output = image_scraper(input_path=image_path, caption=caption, output_path=output_path)

    image_list = ["maoh.jpeg", "jackie-chan.jpg"]
    for image in image_list:
        image_path = os.path.join("./asset", image)
        output_path = f"./{image.split('.')[0]}.md"
        print(f"{image_path} => {output_path}")
        markdown_output = image_scraper(input_path=image_path, caption=caption, output_path=output_path)
        