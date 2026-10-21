import re

def calculate_customer_parameters(email_body: str) -> dict:
    """
    Parses a customer complaint email to extract operational 'hyperparameters'
    such as urgency, sentiment proxy, and churn risk.
    """
    # 1. Base weights and setup
    email_lower = email_body.lower()
    total_words = len(email_lower.split())
    
    # 2. Define operational indicator dictionaries
    high_urgency_words = ['immediately', 'asap', 'broken', 'urgent', 'disaster', 'fail', 'fix']
    churn_risk_words = ['cancel', 'refund', 'competitor', 'sue', 'lawyer', 'leaving', 'stop']
    negative_sentiment_words = ['horrible', 'terrible', 'worst', 'angry', 'hate', 'useless', 'disappointed']
    
    # 3. Calculate keyword occurrences
    urgency_hits = sum(1 for word in high_urgency_words if word in email_lower)
    churn_hits = sum(1 for word in churn_risk_words if word in email_lower)
    negative_hits = sum(1 for word in negative_sentiment_words if word in email_lower)
    
    # 4. Check syntax markers (e.g., ALL CAPS words signal frustration)
    all_caps_words = len(re.findall(r'\b[A-Z]{3,}\b', email_body))
    has_exclamation = 1 if '!' in email_body else 0

    # 5. Compute the final operational 'hyperparameters' (scaled 0.0 to 1.0)
    # Urgency weight increases with explicit terms, length, and exclamation marks
    urgency_weight = min(1.0, (urgency_hits * 0.25) + (has_exclamation * 0.2) + (min(total_words, 200) / 1000))
    
    # Sentiment score proxy (1.0 is perfectly neutral/positive, closer to 0.0 is highly negative)
    sentiment_score = max(0.0, 1.0 - ((negative_hits * 0.2) + (all_caps_words * 0.1)))
    
    # Churn Risk Index based on specific transactional cancellation threats
    churn_risk_index = min(1.0, (churn_hits * 0.35) + (all_caps_words * 0.05))

    return {
        "calculated_urgency_weight": round(urgency_weight, 2),
        "calculated_sentiment_score": round(sentiment_score, 2),
        "calculated_churn_risk_index": round(churn_risk_index, 2),
        "metadata": {
            "word_count": total_words,
            "all_caps_count": all_caps_words
        }
    }

# --- Verification & Example Run ---
sample_email = """
Dear Team,
I am utterly DISAPPOINTED with your service! The interface is broken and completely useless. 
If this is not fixed immediately, I want a full REFUND and will cancel my subscription to move 
to a competitor. ASAP!
"""

customer_hyperparams = calculate_customer_parameters(sample_email)

print("--- Extracted Customer Operational Parameters ---")
for key, value in customer_hyperparams.items():
    print(f"{key}: {value}")
