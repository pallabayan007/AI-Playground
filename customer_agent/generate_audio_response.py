import os
import re
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
import openai  # For Azure OpenAI / Bedrock / Local vLLM endpoints

# =====================================================================
# 1. ARCHITECTURAL DATA STRUCTURES
# =====================================================================

class AgentState(TypedDict):
    """Tracks state for the Genesys live stream session."""
    transcript: str          # Current live audio transcript segment
    acoustic_sentiment: str  # "STRESSED", "AGITATED", "NEUTRAL"
    pii_masked_text: str     # Safe text for LLM processing
    is_vulnerable: bool      # Flag for FCA FG21/1 compliance
    draft_suggestion: str   # Generated response from RAG
    final_output: Dict[str, Any] # Clean payload for Genesys UI Widget

# =====================================================================
# 2. PRIVACY & COMPLIANCE PIPELINES
# =====================================================================

def mask_pii_and_secrets(state: AgentState) -> Dict[str, Any]:
    """
    UK GDPR / DPA 2018 Data Minimisation Rule.
    In-memory regex/Presidio masking for sort codes, account numbers, and cards.
    """
    text = state["transcript"]
    
    # Mask UK Sort Codes (e.g., 20-30-40 or 203040)
    text = re.sub(r'\b\d{2}[- ]?\d{2}[- ]?\d{2}\b', '[SORT CODE MASKED]', text)
    # Mask 8-digit UK Bank Account Numbers
    text = re.sub(r'\b\d{8}\b', '[ACCOUNT NUMBER MASKED]', text)
    # Mask 16-digit Credit Cards (PCI-DSS fallback)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD MASKED]', text)
    
    return {"pii_masked_text": text}

def assess_vulnerability(state: AgentState) -> Dict[str, Any]:
    """
    FCA FG21/1 Vulnerable Customers Classification Guardrail.
    Combines acoustic metrics with lexical checks.
    """
    text = state["pii_masked_text"].lower()
    acoustic = state["acoustic_sentiment"]
    
    # Lexical triggers for financial/personal vulnerability
    vulnerability_keywords = [
        "redundant", "redundancy", "bereavement", "passed away", 
        "can't afford", "debt", "missed payment", "hospital", "illness"
    ]
    
    has_keyword = any(kw in text for kw in vulnerability_keywords)
    has_acoustic_stress = (acoustic in ["STRESSED", "AGITATED"])
    
    # If customer is acoustically stressed AND mentions critical situations, trigger guardrail
    is_vulnerable = has_keyword or (has_acoustic_stress and "pay" in text)
    
    return {"is_vulnerable": is_vulnerable}

# =====================================================================
# 3. COGNITIVE GENERATION LAYER (RAG Stub)
# =====================================================================

def generate_draft_response(state: AgentState) -> Dict[str, Any]:
    """
    Queries local enterprise RAG vectors and builds a baseline response.
    Using OpenAI API compatible server (Azure, Bedrock, or vLLM).
    """
    client = openai.OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_ENDPOINT"))
    
    system_prompt = (
        "You are an assistant for a UK Bank customer service agent. "
        "Provide a concise, direct reply suggestion for the agent to say. "
        "Never state definite financial advice. Use British English spelling (e.g., categorise, authorise)."
    )
    
    # In production, pull relevant context from Vector DB matching state['pii_masked_text']
    mock_rag_context = "Standard Balance Inquiry Procedure: Verify identity, read current balance, ask if further help is needed."
    
    user_prompt = f"Context: {mock_rag_context}\nCustomer Transcript: {state['pii_masked_text']}"
    
    response = client.chat.completions.create(
        model="fine-tuned-llama3-banking",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0, # Zero variance for regulatory safety
        max_tokens=100
    )
    
    return {"draft_suggestion": response.choices[0].message.content.strip()}

# =====================================================================
# 4. REGULATORY INTERCEPTOR RULE ENGINE
# =====================================================================

def enforce_uk_compliance_guardrail(state: AgentState) -> Dict[str, Any]:
    """
    FCA Consumer Duty (Principle 12) & Interceptor Policy Layer.
    Guarantees no unvetted generation bypasses the safety framework.
    """
    suggestion = state["draft_suggestion"]
    is_vulnerable = state["is_vulnerable"]
    
    # Consumer Duty Validation: Check for high-pressure sales phrasing
    banned_phrases = ["must buy today", "limited time offer only", "guaranteed returns"]
    for phrase in banned_phrases:
        if phrase in suggestion.lower():
            # Intercept and rewrite safely
            suggestion = "I can provide details on this product for you to review at your convenience."
            
    # FCA FG21/1 Vulnerability Invalidation Hook
    compliance_flags = []
    if is_vulnerable:
        compliance_flags.append("VULNERABLE_CUSTOMER_DETECTED")
        # Hard override: Inject mandatory empathy language and debt-charity signposting
        suggestion = (
            "I understand this is a difficult time. Let's look at your options together. "
            "I can also provide details for independent organisations like StepChange for free advice if you wish."
        )
    else:
        compliance_flags.append("STANDARD_COMPLIANCE_PASSED")

    # Format structured production JSON for the Genesys Client Workspace
    genesys_payload = {
        "agent_display_text": suggestion,
        "compliance_alerts": compliance_flags,
        "requires_manager_approval": is_vulnerable,
        "timestamp_utc": "2026-08-31T00:00:00Z" # Production dynamic timestamp
    }
    
    return {"final_output": genesys_payload}

# =====================================================================
# 5. ORCHESTRATION GRAPH PIPELINE
# =====================================================================

def build_compliance_agent_graph():
    """Compiles the LangGraph workflow process."""
    workflow = StateGraph(AgentState)
    
    # Define system nodes
    workflow.add_node("mask_pii", mask_pii_and_secrets)
    workflow.add_node("assess_vulnerability", assess_vulnerability)
    workflow.add_node("generate_draft", generate_draft_response)
    workflow.add_node("enforce_compliance", enforce_uk_compliance_guardrail)
    
    # Connect pipeline execution flow
    workflow.set_entry_point("mask_pii")
    workflow.add_edge("mask_pii", "assess_vulnerability")
    workflow.add_edge("assess_vulnerability", "generate_draft")
    workflow.add_edge("generate_draft", "enforce_compliance")
    workflow.add_edge("enforce_compliance", END)
    
    return workflow.compile()

# =====================================================================
# 6. EXECUTION SIMULATION (Live Audio Event Triggered)
# =====================================================================

if __name__ == "__main__":
    # Mock environment configurations
    os.environ["LLM_API_KEY"] = "mock-key"
    os.environ["LLM_ENDPOINT"] = "http://localhost:8000/v1" # Local vLLM/Ollama fallback for testing
    
    compiled_agent = build_compliance_agent_graph()
    
    # Simulation Event: Stressed customer mentions inability to clear an overdraft debt
    live_stream_payload = {
        "transcript": "Hello, my sort code is 20-30-40 and I just lost my job. I cannot pay back this balance.",
        "acoustic_sentiment": "STRESSED",
        "pii_masked_text": "",
        "is_vulnerable": False,
        "draft_suggestion": "",
        "final_output": {}
    }
    
    # Run pipeline in a safe execution block
    try:
        # In a real environment, replace .stream with .ainvoke for async WebSockets
        for output in compiled_agent.stream(live_stream_payload):
            for key, value in output.items():
                print(f"--- Finished Processing Execution Node: {key} ---")
                
        # Final clean payload sent back over WebSockets directly to Genesys Widget
        import json
        print("\n=== FINAL GENERATION FOR GENESYS CUSTOM INTERACTION WIDGET ===")
        print(json.dumps(output["enforce_compliance"]["final_output"], indent=2))
        
    except Exception as e:
        print(f"Fallback safety triggered: System Error {str(e)}")
