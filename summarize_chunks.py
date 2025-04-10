from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def summarize_chunks(texts, tables, images):
    text_prompt = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Table or text chunk: {element}
    """
    text_model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | ChatPromptTemplate.from_template(text_prompt) | text_model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    table_summaries = summarize_chain.batch([t.metadata.text_as_html for t in tables], {"max_concurrency": 3})

    image_prompt = """Describe the image in detail. It’s from a research paper on transformers. Be specific."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": image_prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
            ],
        )
    ]
    img_chain = ChatPromptTemplate.from_messages(messages) | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    image_summaries = img_chain.batch(images)

    return text_summaries, table_summaries, image_summaries
