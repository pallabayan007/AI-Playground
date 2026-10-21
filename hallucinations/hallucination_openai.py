import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# Initialize the OpenAI client (grabs OPENAI_API_KEY from your environment variables)
client = OpenAI()

# ---------------------------------------------------------------------
# 1. Define Data Structures for Structured Outputs
# ---------------------------------------------------------------------
class EvaluatedClaim(BaseModel):
    claim_text: str
    is_supported: bool
    reasoning: str

class ResponseEvaluation(BaseModel):
    all_claims: List[EvaluatedClaim]

# ---------------------------------------------------------------------
# 2. Logic to Compute Scores
# ---------------------------------------------------------------------
def evaluate_response_hallucination(retrieved_context: str, generated_response: str) -> float:
    """
    Uses OpenAI to extract individual claims and grade them against the context.
    Returns a score from 0.0 (Perfectly Factual) to 1.0 (Completely Hallucinated).
    """
    
    system_prompt = (
        "You are an expert AI evaluator. Your job is to detect hallucinations by checking "
        "if a Generated Response is fully grounded in the provided Source Context.\n\n"
        "Instructions:\n"
        "1. Break the Generated Response down into individual, atomic factual claims.\n"
        "2. For each claim, determine if it is directly supported by the Source Context (is_supported=true) "
        "or if it is unsupported/contradicted/fabricated (is_supported=false).\n"
        "3. Provide brief reasoning for each verdict."
    )
    
    user_prompt = f"""
    [Source Context]
    {retrieved_context}
    
    [Generated Response]
    {generated_response}
    """

    # Call OpenAI using Structured Outputs to ensure rigid JSON parsing
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini", # Cost-effective and highly precise for evaluation tasks
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ResponseEvaluation,
        temperature=0.0 # Force maximum determinism
    )
    
    # Extract the structured response object
    evaluation_result = completion.choices[0].message.parsed
    
    total_claims = len(evaluation_result.all_claims)
    if total_claims == 0:
        return 0.0 # No factual claims were made (e.g., "I don't know.")
        
    supported_claims = sum(1 for claim in evaluation_result.all_claims if claim.is_supported)
    
    # Calculate response-level Hallucination Score (Atomic Precision)
    factual_precision = supported_claims / total_claims
    hallucination_score = 1.0 - factual_precision
    
    # Print out visual breakdown for transparency
    print(f"\n--- Claim Evaluation Breakdown ({total_claims} found) ---")
    for idx, c in enumerate(evaluation_result.all_claims, 1):
        status = "✅ PASSED" if c.is_supported else "❌ HALLUCINATION"
        print(f" Claim {idx}: {c.claim_text}")
        print(f" Status: {status} | Reason: {c.reasoning}")
        
    return round(hallucination_score, 2)

def calculate_dataset_hallucination_rate(scores: List[float], threshold: float = 0.30) -> float:
    """
    Calculates the aggregate hallucination rate across the test dataset.
    Any response with more than 30% hallucinated content is flagged.
    """
    if not scores:
        return 0.0
    
    hallucinated_count = sum(1 for score in scores if score > threshold)
    hallucination_rate = (hallucinated_count / len(scores)) * 100
    return round(hallucination_rate, 2)

# ---------------------------------------------------------------------
# 3. Running the Pipeline Evaluation Example
# ---------------------------------------------------------------------

# A mock test dataset representing a RAG system evaluation pipeline
test_dataset = [
    {
        "id": "Test_Case_1",
        "context": "ACME Corp was founded in 1995 by Jane Doe. It sells software tools.",
        "response": "ACME Corp was started in 1995 by Jane Doe. They create software applications."
    },
    {
        "id": "Test_Case_2",
        "context": "Python 3.12 was released in October 2023. It introduced isolated subinterpreters.",
        "response": "Python 3.12 came out in October 2023. It features isolated subinterpreters and was designed by Bjarne Stroustrup."
    }
]

# Step A: Loop through dataset to collect response-level scores
collected_scores = []

for case in test_dataset:
    print(f"\n================ Evaluating {case['id']} ================")
    h_score = evaluate_response_hallucination(case["context"], case["response"])
    collected_scores.append(h_score)
    print(f">> Resulting Hallucination Score: {h_score}")

# Step B: Calculate overall dataset metrics
THRES = 0.30 # Quality gate threshold 
final_rate = calculate_dataset_hallucination_rate(collected_scores, threshold=THRES)

print("\n================ FINAL METRICS ================")
print(f"All Computed Scores: {collected_scores}")
print(f"Dataset Hallucination Rate: {final_rate}% (Threshold: > {THRES})")
