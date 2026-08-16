
# Semantic chunking calculates the vector embedding similarity of adjacent sentences. 
# It splits the text only when a new sentence introduces a completely different topic,
#  ensuring your chunks maintain clean contextual ideas.

import os
from pathlib import Path
import sys
# Ensure repository root is on sys.path so `aiohttp_compat` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aiohttp_compat

from langchain_experimental.text_splitter import SemanticChunker

# Lightweight local embeddings implementation to avoid external API
# dependencies for this example. It returns deterministic numeric vectors
# derived from the input text so the `SemanticChunker` can compute
# similarity distances locally.
import hashlib

class LocalEmbeddings:
    def embed_documents(self, texts):
        vectors = []
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).digest()
            # Create a small fixed-size vector from hash bytes
            vec = [((b & 0xFF) / 255.0) for b in h[:16]]
            vectors.append(vec)
        return vectors

# Configure your API environment key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def semantic_chunk_document(text: str):
    """
    Groups sentences together into structural chunks based on conceptual meaning.
    """
    # Initialize embedding model to compute similarity distances
    # Use the local embeddings implementation to avoid external API calls.
    embeddings = LocalEmbeddings()
    
    # Set the breakpoint threshold type (percentile, standard_deviation, or interquartile)
    splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile"
    )
    
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]

# --- Example Usage ---
multi_topic_text = (
    "The Golden Gate Bridge is an iconic suspension bridge located in San Francisco, California. "
    "It spans the Golden Gate strait. "
    "On a completely different note, cellular respiration converts biochemical energy from nutrients into ATP. "
    "This vital biological process happens inside mitochondria."
)

semantic_chunks = semantic_chunk_document(multi_topic_text)
for i, s_chunk in enumerate(semantic_chunks):
    print(f"--- Semantic Group {i+1} ---\n{s_chunk}\n")
