from typing import Optional
import base64
import os
from openai import OpenAI


class ImageToMarkdownScraper:
    def __init__(self, OPENAI_API_KEY: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def __call__(self, image_path: str, caption: Optional[str] = None, output_path: Optional[str] = None) -> str:
        # Validate image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the message for the API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the content of this image in detail. Format your response in markdown, including appropriate headers and bullet points where relevant."
                    },
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
            messages[0].content.append({"type": "text", "text": caption})
        

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300
        )

        # Extract the markdown description
        markdown_description = response.choices[0].message.content

        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as md_file:
                md_file.write(markdown_description)
            print(f"Markdown description saved to: {output_path}")

        return markdown_description

# Usage example
if __name__ == "__main__":
    OPENAI_API_KEY = ""
    model = "gpt-4o-mini-2024-07-18"
    image_scraper = ImageToMarkdownScraper(OPENAI_API_KEY, model)
    
    # for image in os.listdir("./asset"):
    #     if image.split('.')[1] != 'md':
    #         image_path = os.path.join("./asset", image)
    #         output_path = image_path.replace(".jpg", ".md")
    #         markdown_output = image_scraper(image_path, image, output_path)

    image_path = "./asset/tzuchi-activity.jpg"
    caption = """
校長教師愛灑 有趣學習更有效 | 新聞
https://www.facebook.com/DaAiChannelMalaysia.chn/videos/547851927940870

慈濟教育團隊在藍毘尼魯潘德希縣（Rupandehi District）的悉達多市（Bhairahawa）和布德沃爾（Butwal）舉辦私立學校校長和教師愛灑，傳授五i教學法，把學堂變得有趣，學生更有興趣上學去！

YouTube: https://youtu.be/nHThzKoeC2o
    """
    
    output_path = image_path.replace(".jpg", ".md")
    markdown_output = image_scraper(image_path, caption, output_path)
