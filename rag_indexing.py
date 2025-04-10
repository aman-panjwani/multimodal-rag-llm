def build_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries):
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain.storage import InMemoryStore
    from langchain.schema.document import Document
    import uuid
    from langchain.retrievers.multi_vector import MultiVectorRetriever

    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

    def docs_with_ids(data, summaries):
        docs, originals = [], []
        for i, (item, summary) in enumerate(zip(data, summaries)):
            if not summary or not summary.strip():
                continue  # Skip empty summaries
            doc_id = str(uuid.uuid4())
            docs.append(Document(page_content=summary, metadata={"doc_id": doc_id}))
            originals.append((doc_id, item))
        return docs, originals

    # Add documents if summaries are valid
    for content, summaries in zip(
        [texts, tables, images],
        [text_summaries, table_summaries, image_summaries]
    ):
        docs, originals = docs_with_ids(content, summaries)
        if docs:
            retriever.vectorstore.add_documents(docs)
            retriever.docstore.mset(originals)

    return retriever
