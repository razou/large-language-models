import os
import openai
from dotenv import load_dotenv, find_dotenv


class PrepareEnv:
    def set_api_key():
        _ = load_dotenv(find_dotenv())
        openai.api_key = os.environ["OPENAI_API_KEY"]