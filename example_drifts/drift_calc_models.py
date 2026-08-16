import numpy as np
from pathlib import Path
import sys
# Ensure repository root is on sys.path so `aiohttp_compat` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aiohttp_compat

from openai import OpenAI
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# 1. Initialize clients
openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
claude_client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_openai_response(prompt: str) -> str:
  response = openai_client.chat.completions.create(
    #   model="gpt-4o",
      model="gpt-5.6-luna",
      messages=[{"role": "user", "content": prompt}],
    #   temperature=0.0,
      temperature=1,
  )
  return response.choices[0].message.content


def get_claude_response(prompt: str) -> str:
  response = claude_client.messages.create(
      model="claude-fable-5",
      max_tokens=1024,
      messages=[{"role": "user", "content": prompt}],
  )
  return response.content[0].text


# 2. Run test prompt through both models
test_prompt = "Explain the trade-offs of microservices vs monoliths in two sentences."

openai_text = get_openai_response(test_prompt)
claude_text = get_claude_response(test_prompt)

# 3. Embed the responses into semantic vectors
embeddings = embedder.encode([openai_text, claude_text])
openai_emb = embeddings[0].reshape(1, -1)
claude_emb = embeddings[1].reshape(1, -1)

# 4. Calculate semantic drift (cosine distance: 0 = identical, 2 = opposite)
drift_score = cosine_distances(openai_emb, claude_emb)[0][0]

print(f"OpenAI Output: {openai_text}\n")
print(f"Claude Output: {claude_text}\n")
print(f"Semantic Drift Score (Cosine Distance): {drift_score:.4f}")
