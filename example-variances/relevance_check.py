import os
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# 1. Set your API key (DeepEval defaults to OpenAI, but supports others)
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 2. Define the user query and the LLM's generated response
user_input = "What is the capital of France?"
llm_output = "France is a country in Europe. It is known for fashion, art, and the Eiffel Tower."

# 3. Initialize the relevance metric (threshold determines pass/fail)
relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")

# 4. Create an evaluation test case
test_case = LLMTestCase(
    input=user_input,
    actual_output=llm_output
)

# 5. Execute the metric evaluation
relevancy_metric.measure(test_case)

# 6. Print the results
print(f"Relevance Score: {relevancy_metric.score}")  # Score from 0.0 to 1.0
print(f"Is Relevant?: {relevancy_metric.is_successful()}")
print(f"Reasoning for score:\n{relevancy_metric.reason}")
