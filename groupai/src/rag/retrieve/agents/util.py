import os
import openai


class Encoder:
    def __init__(
            self, embedding_model: str = "text-embedding-3-small",
    ):
        self.__embedding_model = embedding_model

    def text_to_embedding(self, text: str):
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=text, model=self.__embedding_model)
            return response.data[0].embedding
        except Exception as e:
            print(f"text_to_embedding: {e}")
            raise
