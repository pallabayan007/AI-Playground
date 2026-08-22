from openai import OpenAI

# client = OpenAI(api_key="YOUR_API_KEY")
client = OpenAI()

def generate_support_draft(customer_name, history, sentiment, complaint, context):
    prompt = f"""
    You are a helpful customer support agent. Write a professional email response.
    
    Customer Name: {customer_name}
    Interaction History: {history}
    Current Sentiment: {sentiment}
    Complaint Content: {complaint}
    Business Context/Policy: {context}
    
    Draft a polite, empathetic, and clear response addressing the complaint.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        top_p=0.95,
        # top_k=40,
        # max_output_tokens=500
    )
    
    return response.choices[0].message.content

# Example usage:
draft = generate_support_draft(
    customer_name="Alice",
    history="Ordered a laptop on Aug 10. Delayed shipping.",
    sentiment="Frustrated",
    complaint="Where is my order? It is two weeks late.",
    context="Offer a 10% refund on shipping and state it will arrive in 2 days."
)
print(draft)
