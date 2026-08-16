import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def check_groundedness(context: str, response: str) -> float:
    """Calculates a groundedness score between 0 and 1 using an NLI model."""
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Format the input for NLI (Premise: Context, Hypothesis: LLM Response)
    inputs = tokenizer(
        context, response, return_tensors="pt", truncation=True, max_length=1024
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and apply softmax to get probabilities
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).tolist()[0]

    # BART-large-mnli labels: 0 = contradiction, 1 = neutral, 2 = entailment
    entailment_score = probs[2]
    return entailment_score


# --- Example Usage ---
reference_context = (
    "The Solar System consists of the Sun and the objects that orbit it."
)

# good_response = "The Sun is at the center of the Solar System."
# good_response = "The Sun is within the Solar System."
good_response = "The Solar System consists of the Sun and the objects that orbit it."S
bad_response = "The Solar System has exactly fourteen distinct planets."

print(f"Grounded Score (Valid): {check_groundedness(reference_context, good_response):.4f}")
print(f"Grounded Score (Hallucinated): {check_groundedness(reference_context, bad_response):.4f}")
