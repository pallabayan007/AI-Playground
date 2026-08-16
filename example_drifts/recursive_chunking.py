
# This technique preserves structural text layouts (like paragraphs and sentences) by trying a hierarchy of delimiters (\n\n, \n,  , "") 
# before breaking content by strict length limitations

from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_chunk_document(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits text recursively based on structure (paragraphs, sentences).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,        # Maximum character count per chunk
        chunk_overlap=chunk_overlap,  # Shared characters between consecutive chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process text directly into a list of strings
    chunks = splitter.split_text(text)
    return chunks

# --- Example Usage ---
document_text = """
Artificial Intelligence (AI) is transforming industries globally. By automating complex routines, 
companies increase efficiency. This shift fundamentally alters workforce dynamics.

However, scaling AI systems demands robust pipelines. Clean data engineering remains the true bottleneck 
for most enterprises trying to scale Large Language Models.
"""

chunks = recursive_chunk_document(document_text, chunk_size=120, chunk_overlap=20)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}\n")
