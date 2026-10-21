def calculate_hallucination_score(verified_claims: int, total_claims: int) -> float:
    """
    Calculates the hallucination score for a single model response.
    Score ranges from 0.0 (Perfectly Factual) to 1.0 (Completely Hallucinated).
    """
    if total_claims == 0:
        return 0.0  # Avoid division by zero if response has no factual claims
        
    factual_precision = verified_claims / total_claims
    hallucination_score = 1.0 - factual_precision
    
    return round(hallucination_score, 2)


def calculate_hallucination_rate(scores: list[float], threshold: float = 0.40) -> float:
    """
    Calculates the overall dataset hallucination rate.
    Any score strictly greater than the threshold is flagged as a hallucination.
    """
    if not scores:
        return 0.0
        
    # Count how many responses crossed the unacceptable hallucination threshold
    hallucinated_responses = sum(1 for score in scores if score > threshold)
    
    # Calculate the percentage rate across all eligible test cases
    hallucination_rate = (hallucinated_responses / len(scores)) * 100
    
    return round(hallucination_rate, 2)


# =====================================================================
# SIMULATED PIPELINE RUN
# =====================================================================

# Mock evaluation data parsed from an LLM-as-a-judge or human audit
eval_dataset = [
    {"prompt_id": 101, "total_claims": 5, "verified_claims": 5}, # Perfectly factual
    {"prompt_id": 102, "total_claims": 4, "verified_claims": 3}, # 1 minor error (Score 0.25)
    {"prompt_id": 103, "total_claims": 3, "verified_claims": 1}, # Heavily hallucinated (Score 0.67)
    {"prompt_id": 104, "total_claims": 6, "verified_claims": 2}, # Heavily hallucinated (Score 0.67)
    {"prompt_id": 105, "total_claims": 2, "verified_claims": 2}, # Perfectly factual
]

# 1. Process individual records to gather Hallucination Scores
all_scores = []
print("--- Response Level Breakdown ---")
for record in eval_dataset:
    score = calculate_hallucination_score(record["verified_claims"], record["total_claims"])
    all_scores.append(score)
    print(f"Prompt {record['prompt_id']}: Hallucination Score = {score}")

# 2. Aggregating dataset to calculate total Hallucination Rate
# We set the tolerance threshold at 0.40 (responses with >40% fake info are flagged)
QUALITY_THRESHOLD = 0.40
final_rate = calculate_hallucination_rate(all_scores, threshold=QUALITY_THRESHOLD)

print("\n--- Dataset Level Summary ---")
print(f"Collected Scores: {all_scores}")
print(f"Quality Gate Threshold: {QUALITY_THRESHOLD}")
print(f"Final Hallucination Rate: {final_rate}%")
