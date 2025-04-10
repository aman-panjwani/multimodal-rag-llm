import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

