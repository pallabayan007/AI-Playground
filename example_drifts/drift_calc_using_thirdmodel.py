import json
import numpy as np
from openai import OpenAI
import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# 1. Initialize API Clients and Local Embedder
openai_client = OpenAI()  # Uses OPENAI_API_KEY
claude_client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_openai_response(prompt: str) -> str:
    """Fetches response from OpenAI GPT-4o."""
    response = openai_client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": prompt}],
        # temperature=1,
    )
    return response.choices[0].message.content


def get_claude_response(prompt: str) -> str:
    """Fetches response from Anthropic Claude 3.5 Sonnet."""
    response = claude_client.messages.create(
        model="claude-fable-5",
        max_tokens=1024,
        # temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def evaluate_with_judge(prompt: str, response_a: str, response_b: str) -> dict:
    """Uses GPT-4o as a structured judge to evaluate functional correctness."""
    judge_system_prompt = (
        "You are an expert AI evaluator. Your task is to compare two AI model responses (A and B) "
        "against an initial user prompt. Evaluate them for functional correctness, factual accuracy, "
        "and adherence to instructions.\n\n"
        "Provide your output strictly in JSON format with two keys:\n"
        "1. 'correctness_score': An integer from 1 to 5 where:\n"
        "   - 5: Both models are perfectly correct and follow all constraints.\n"
        "   - 4: Both are functional, but one misses a minor detail or formatting request.\n"
        "   - 3: One model is correct, but the other contains an error, hallucination, or failure.\n"
        "   - 2: Both models fail major constraints or show minor functional errors.\n"
        "   - 1: Both models are functionally broken, incorrect, or completely ignore constraints.\n"
        "2. 'reasoning': A short 2-sentence explanation of why you gave that score, highlighting differences."
    )

    judge_user_prompt = f"""
    User Prompt: {prompt}
    
    Response A (OpenAI):
    {response_a}
    
    Response B (Claude):
    {response_b}
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": judge_system_prompt},
                {"role": "user", "content": judge_user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"correctness_score": 0, "reasoning": f"Judge error: {str(e)}"}


# =====================================================================
# Execution Pipeline
# =====================================================================

# Test prompt containing specific structural and factual constraints
test_prompt = (
    "Write a Python function named 'is_prime' that returns a boolean. "
    "Do not use any loops; use recursion instead. Provide only the code block."
)

print("Fetching model outputs...")
openai_out = get_openai_response(test_prompt)
claude_out = get_claude_response(test_prompt)

# Calculate Semantic Drift (Cosine Distance)
# Reshape arrays to 2D for scikit-learn compatibility
embs = embedder.encode([openai_out, claude_out])
drift_score = cosine_distances(embs[0].reshape(1, -1), embs[1].reshape(1, -1))[0][0]

# Run LLM Evaluation
print("Evaluating outputs with LLM Judge...")
judge_evaluation = evaluate_with_judge(test_prompt, openai_out, claude_out)

# Print Final Evaluation Dashboard
print("\n" + "="*50)
print("EVALUATION METRICS DASHBOARD")
print("="*50)
print(f"Semantic Drift (0.00 = Identical, 2.00 = Opposite): {drift_score:.4f}")
print(f"Judge Correctness Score (1-5 Scale): {judge_evaluation.get('correctness_score')}/5")
print(f"Judge Reasoning: {judge_evaluation.get('reasoning')}")
print("-"*50)
print(f"--- OPENAI OUTPUT ---\n{openai_out}\n")
print(f"--- CLAUDE OUTPUT ---\n{claude_out}")
print("="*50)
