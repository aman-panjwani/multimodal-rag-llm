from env_setup import load_env
from pdf_processing import process_pdf
from summarize_chunks import summarize_chunks
from rag_indexing import build_retriever
from query_chain import get_chain
import os

def execute_main():
    print("[Step:1] Loading environment...")
    load_env()
    
    print("[Step:2] Processing PDF...")
    texts, tables, images = process_pdf()

    print(f"Extracted: {len(texts)} text chunks, {len(tables)} tables, {len(images)} images")

    print("[Step:3] Summarizing content...")
    text_summaries, table_summaries, image_summaries = summarize_chunks(texts, tables, images)

    print("[Step:4] Indexing data...")
    retriever = build_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries)

    print("[Step:5] Ready to answer questions!")
    chain = get_chain(retriever)

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.strip().lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break
        result = chain.invoke(question)
        print("Answer:", result)

if __name__ == "__main__":
    execute_main()
