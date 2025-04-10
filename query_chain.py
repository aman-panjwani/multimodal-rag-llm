from base64 import b64decode
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs = kwargs["context"]
    question = kwargs["question"]

    context_text = "".join([t.text for t in docs["texts"]])
    prompt_text = f"""
    Answer the question based only on the context below (text, tables, images).
    Context: {context_text}
    Question: {question}
    """
    prompt = [{"type": "text", "text": prompt_text}]
    prompt += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in docs["images"]]

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])

def get_chain(retriever):
    return {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnableLambda(build_prompt) | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
