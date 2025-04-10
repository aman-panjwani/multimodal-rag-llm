from unstructured.partition.pdf import partition_pdf

def process_pdf(file_path="./content/AI-Intro.pdf"):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    tables, texts = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)

    return texts, tables, images
