# LLMs do not read characters; they read tokens. 
# Splitting by a token length limit guarantees that your processed chunks will fit neatly 
# inside embedding or completion context limits without triggering an out-of-bounds error.

import tiktoken

def token_chunk_document(text: str, model_name: str = "gpt-4", chunk_size: int = 50, chunk_overlap: int = 10):
    """
    Chunks document text by exact LLM token counts using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = start_idx + chunk_size
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Decode tokens back into readable string content
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Advance window incorporating the targeted overlap space
        start_idx += (chunk_size - chunk_overlap)
        
        if end_idx >= len(tokens):
            break
            
    return chunks

# --- Example Usage ---
sample_data = "Data processing pipelines require extraction, transformation, layout chunking, and indexing."
token_chunks = token_chunk_document(sample_data, chunk_size=8, chunk_overlap=2)

for i, t_chunk in enumerate(token_chunks):
    print(f"Token Chunk {i+1}: '{t_chunk}'")
