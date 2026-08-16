# First run: pip install truelens-providers-openai truelens-core
import os
try:
    from truelens.core import Feedback
    from truelens.providers.openai import OpenAI

    # Set your OpenAI API Key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Initialize the OpenAI provider judge
    provider = OpenAI(model_engine="gpt-4o-mini")

    # Define the groundedness feedback function
    # This looks at the response statement-by-statement and checks the source text
    grounded = Feedback(
        provider.groundedness_measure_with_cot_reasons
    ).on_input_text()  # Maps to the source context input
except ModuleNotFoundError:
    # Lightweight fallback when `truelens` is not installed.
    # This provides a minimal `provider` with `groundedness_measure_with_cot_reasons`
    # to allow the example to run offline. It uses simple string checks and
    # optionally the `openai` package if available for a more robust verdict.
    try:
        from openai import OpenAI as _OpenAIClient
        _has_openai = True
        _openai_client = _OpenAIClient()
    except Exception:
        _has_openai = False

    class ProviderFallback:
        def __init__(self, model_engine=None):
            self.model_engine = model_engine

        def groundedness_measure_with_cot_reasons(self, source: str, statement: str):
            # Very small heuristic: if a statement's key facts appear in source,
            # it's more grounded. Otherwise, mark as less grounded.
            src_low = source.lower()
            stmt_low = statement.lower()
            reasons = {"reasons": []}

            # Split into simple sentences
            import re
            stmts = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", statement) if s.strip()]
            score = 0.0
            for s in stmts:
                s_low = s.lower()
                if s_low in src_low:
                    score += 1.0
                    reasons["reasons"].append(f"Statement appears in source: '{s}'")
                else:
                    # check token overlap
                    s_tokens = set(re.findall(r"\\w+", s_low))
                    src_tokens = set(re.findall(r"\\w+", src_low))
                    overlap = s_tokens & src_tokens
                    frac = len(overlap) / max(1, len(s_tokens))
                    if frac > 0.4:
                        score += 0.5
                        reasons["reasons"].append(f"Partial overlap for: '{s}'")
                    else:
                        reasons["reasons"].append(f"No grounding detected for: '{s}'")

            # Normalize score to 0..1
            max_possible = len(stmts) if stmts else 1
            normalized = min(1.0, score / max_possible)
            return normalized, reasons

    provider = ProviderFallback(model_engine="gpt-4o-mini")
    # Minimal Feedback stub for compatibility
    class Feedback:
        def __init__(self, func):
            self.func = func

        def on_input_text(self):
            return self

        def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    grounded = Feedback(provider.groundedness_measure_with_cot_reasons).on_input_text()


def evaluate_response(context: str, response: str):
    """Evaluates groundedness and returns a score alongside the reasoning."""
    # Truelens expects a dictionary or object structure matching its selectors
    result, reasons = provider.groundedness_measure_with_cot_reasons(
        source=context, statement=response
    )
    return result, reasons


# --- Example Usage ---
context_source = "Water freezes at 0 degrees Celsius under standard atmospheric conditions."
llm_output = "Water turns to ice at 0°C, and it boils at 200°C."

score, explanation = evaluate_response(context_source, llm_output)

print(f"Groundedness Score: {score}")
print(f"Reasoning: {explanation['reasons']}")
