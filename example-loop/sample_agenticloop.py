import json
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# 1. Define Dummy Tools (The Agent's "Hands")
def get_current_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    if "london" in location.lower():
        return json.dumps({"temperature": "15", "unit": "C", "condition": "Rainy"})
    elif "los angeles" in location.lower():
        return json.dumps({"temperature": "28", "unit": "C", "condition": "Sunny"})
    return json.dumps({"temperature": "unknown"})

# Map tool names to their actual Python executable functions
available_tools = {
    "get_current_weather": get_current_weather
}

# Define the JSON Schema for the tool so the LLM knows how to call it
tools_specification = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA or London, UK",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

# 2. The Core Agentic Loop Architecture
def run_agentic_loop(user_prompt: str, max_iterations: int = 5):
    print(f"🚀 Goal: {user_prompt}\n")
    
    # Initialize conversational memory (State)
    messages = [
        {"role": "system", "content": "You are an autonomous helper. Use tools when necessary to solve the user's request."},
        {"role": "user", "content": user_prompt}
    ]
    
    # The Loop: Observe -> Decide -> Act -> Evaluate
    for iteration in range(1, max_iterations + 1):
        print(f"--- [Iteration {iteration}] Evaluating next step... ---")
        print(f"📝 Current Memory: {messages}\n")
        
        # Step 1: Consult the LLM (The "Brain")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_specification
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message) # Append assistant decision to history
        print(f"🤖 Assistant's Decision: {assistant_message.content}")
        print(f"🛠️ Tool Calls: {assistant_message.tool_calls}\n")
        print(f"📜 Full Assistant Message: {assistant_message}\n")
        
        # Step 2: Evaluation Check - Did it finish or does it need tools?
        if not assistant_message.tool_calls:
            print("\n🏁 Goal Achieved! Final response from agent:")
            print(assistant_message.content)
            return
        
        # Step 3: Execution - Loop through and run requested tools
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"🔧 Action: LLM invoked '{function_name}' with args: {function_args}")
            
            # Lookup and execute tool safely
            tool_function = available_tools.get(function_name)
            if tool_function:
                tool_output = tool_function(**function_args)
            else:
                tool_output = json.dumps({"error": f"Tool {function_name} not found."})
                
            print(f"📥 Observation: Tool returned -> {tool_output}\n")
            
            # Feed the tool execution outcome back to the LLM's memory
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": tool_output
            })
            
    print(f"⚠️ Warning: Agent hit the maximum iteration limit ({max_iterations}) without completing.")

# 3. Trigger the execution
if __name__ == "__main__":
    query = "Compare the current weather in London and Los Angeles, then summarize if I should pack an umbrella for both."
    run_agentic_loop(query)
