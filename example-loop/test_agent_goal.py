import abc
from typing import Dict, Any, Tuple

# 1. Define the Environment State
class Environment:
    def __init__(self):
        # Simulated buggy state
        self.files = {"auth.py": "def login(): return False # BUG: Always returns False"}
        self.test_suite_passing = False

    def run_tests(self) -> Tuple[int, str]:
        """Deterministic verification tool."""
        if "return True" in self.files["auth.py"]:
            self.test_suite_passing = True
            return 0, "All tests passed successfully."
        return 1, "AssertionError: Expected True, got False."

    def write_file(self, filename: str, content: str):
        self.files[filename] = content


# 2. Define the Agent Contract
class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def clean_thought_loop(self, last_observation: str) -> Dict[str, Any]:
        """Agent reasons and decides on a tool action."""
        pass


class MockDeveloperAgent(BaseAgent):
    def __init__(self):
        self.attempts = 0

    def clean_thought_loop(self, last_observation: str) -> Dict[str, Any]:
        self.attempts += 1
        if self.attempts == 1:
            return {
                "thought": "The tests are failing. I need to inspect and rewrite auth.py.",
                "action": "write_file",
                "args": {"filename": "auth.py", "content": "def login(): return True # Fixed"}
            }
        return {"thought": "Tests should be passing now. I will check.", "action": "none", "args": {}}


# 3. The Agentic Loop Testing Harness
class AgenticLoopTester:
    def __init__(self, agent: BaseAgent, env: Environment, max_iterations: int = 5):
        self.agent = agent
        self.env = env
        self.max_iterations = max_iterations
        self.trace_logs = []

    def execute_goal_test(self) -> Dict[str, Any]:
        iteration = 0
        last_observation = "Initial State: Run tests to find bugs."
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Step A: Reason & Act
            agent_decision = self.agent.clean_thought_loop(last_observation)
            action = agent_decision.get("action")
            args = agent_decision.get("args", {})
            
            # Step B: Apply Actions to Environment
            if action == "write_file":
                self.env.write_file(args["filename"], args["content"])
            
            # Step C: Observe & Evaluate (Independent Verification)
            exit_code, stdout = self.env.run_tests()
            last_observation = f"Exit Code: {exit_code}. Output: {stdout}"
            
            # Log trace for assertion analysis
            self.trace_logs.append({
                "iteration": iteration,
                "thought": agent_decision["thought"],
                "action": action,
                "observation": last_observation
            })
            
            # Goal Check: Independent verification criteria met?
            if exit_code == 0:
                return {
                    "success": True, 
                    "iterations": iteration, 
                    "reason": "Goal reached cleanly.",
                    "traces": self.trace_logs
                }
                
        return {
            "success": False, 
            "iterations": iteration, 
            "reason": "Max iterations reached without satisfying goal.",
            "traces": self.trace_logs
        }


# 4. Asserting Goal Criteria in Pytest/Unit Tests
def test_agent_goal_convergence():
    env = Environment()
    agent = MockDeveloperAgent()
    loop_tester = AgenticLoopTester(agent=agent, env=env, max_iterations=3)
    
    # Run the system loop
    result = loop_tester.execute_goal_test()
    
    # Assertions validating the Goal and Stop-Conditions
    assert result["success"] is True, f"Agent failed to complete goal: {result['reason']}"
    assert result["iterations"] <= 3, "Agent took too many iterations to converge."
    assert "return True" in env.files["auth.py"], "Goal mutated state incorrectly."
    print("\n✅ Goal verified successfully via automated test harness.")

if __name__ == "__main__":
    test_agent_goal_convergence()
