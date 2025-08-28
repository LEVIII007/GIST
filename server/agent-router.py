from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from .spam_classifier import SpamClassifier
from .task_extractor import TaskExtractor
from .content_classifier import ContentClassifier
from .content_summarizer import ContentSummarizer
from .personality_summarizer import PersonalitySummarizer
from .domain_inference_agent import DomainInferenceAgent
from .email_extractor import EmailExtractorAgent
from .email_scorer import EmailScorerAgent
from .questions_generator import QuestionsGenerator
from .feedback_learning_agent import FeedbackLearningAgent
from .task_cost_features_extractor import CostFeaturesExtractor
from .task_utility_features_extractor import UtilityFeaturesExtractor
from langfuse.decorators import observe
import json

class AgentRouter(BaseAgent):
    """
    Intelligent router that determines which agent should handle a user request.
    Uses LLM to analyze the request and route to the appropriate specialized agent.
    """

    def __init__(self):
        super().__init__()
        self.routing_prompt = """You are an intelligent agent router. Your job is to analyze user requests and determine which specialized agent should handle them.

Available agents and their capabilities:

1. **spam_classifier**: Classifies emails as spam or not spam
   - Use for: "Is this email spam?", "Check if this is spam", "Spam detection"
   - Input: Email content/text

Analyze the user request and respond with:
1. The most appropriate agent name
2. A brief explanation of why this agent is the best choice
3. Any preprocessing needed for the input

Respond in JSON format:
{
    "agent": "agent_name",
    "explanation": "Why this agent is best",
    "preprocessing": "Any input modifications needed",
    "confidence": 0.95
}"""

        # Initialize all available agents
        self.agents = {
            "spam_classifier": SpamClassifier(),
            "task_extractor": TaskExtractor(),
            "content_classifier": ContentClassifier(),
            "content_summarizer": ContentSummarizer(),
            "personality_summarizer": PersonalitySummarizer(),
            "domain_inference": DomainInferenceAgent(),
            "email_extractor": EmailExtractorAgent(),
            "email_scorer": EmailScorerAgent(),
            "questions_generator": QuestionsGenerator(),
            "feedback_learning": FeedbackLearningAgent(),
            "cost_features_extractor": CostFeaturesExtractor(),
            "utility_features_extractor": UtilityFeaturesExtractor(),
        }

    @observe()
    async def route_request(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a user request to the appropriate agent.
        
        Args:
            user_request: The user's request text
            context: Optional context information (user_id, domain, etc.)
            
        Returns:
            Dict containing routing decision and agent response
        """
        try:
            # Step 1: Analyze the request to determine routing
            routing_decision = await self._analyze_request(user_request, context)
            
            if not routing_decision or "agent" not in routing_decision:
                return {
                    "error": "Failed to determine appropriate agent",
                    "user_request": user_request
                }
            
            agent_name = routing_decision["agent"]
            
            # Step 2: Validate agent exists
            if agent_name not in self.agents:
                return {
                    "error": f"Unknown agent: {agent_name}",
                    "available_agents": list(self.agents.keys()),
                    "routing_decision": routing_decision
                }
            
            # Step 3: Get the appropriate agent
            agent = self.agents[agent_name]
            
            # Step 4: Preprocess input if needed
            processed_input = self._preprocess_input(user_request, routing_decision, context)
            
            # Step 5: Execute the agent
            agent_response = await self._execute_agent(agent, agent_name, processed_input, context)
            
            return {
                "routing_decision": routing_decision,
                "agent_used": agent_name,
                "agent_response": agent_response,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Routing failed: {str(e)}",
                "user_request": user_request,
                "success": False
            }

    async def _analyze_request(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to analyze the request and determine routing."""
        
        # Prepare the analysis prompt
        analysis_prompt = f"""User Request: {user_request}

Context: {json.dumps(context) if context else 'None'}

Please analyze this request and determine which agent should handle it."""

        try:
            response = await self.execute(
                system_prompt=self.routing_prompt,
                user_input=analysis_prompt,
                response_format="json"
            )
            
            if isinstance(response, dict):
                return response
            else:
                # Fallback parsing if response is string
                try:
                    return json.loads(response)
                except:
                    return {"agent": "content_classifier", "explanation": "Fallback routing", "confidence": 0.5}
                    
        except Exception as e:
            print(f"Routing analysis failed: {e}")
            # Fallback to content classifier for general requests
            return {
                "agent": "content_classifier",
                "explanation": "Fallback routing due to analysis failure",
                "confidence": 0.3
            }

    def _preprocess_input(self, user_request: str, routing_decision: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """Preprocess input based on routing decision."""
        
        preprocessing = routing_decision.get("preprocessing", "")
        agent_name = routing_decision.get("agent", "")
        
        # Agent-specific preprocessing
        if agent_name == "spam_classifier":
            # Extract email content from request
            if "email:" in user_request.lower() or "content:" in user_request.lower():
                # Try to extract the actual content
                parts = user_request.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
            return user_request
            
        elif agent_name == "task_extractor":
            # Extract task-related content
            if "email:" in user_request.lower():
                parts = user_request.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
            return user_request
            
        elif agent_name == "content_summarizer":
            # Extract content to summarize
            if "summarize:" in user_request.lower() or "content:" in user_request.lower():
                parts = user_request.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
            return user_request
            
        return user_request

    async def _execute_agent(self, agent, agent_name: str, processed_input: Any, context: Optional[Dict[str, Any]]) -> Any:
        """Execute the selected agent with the processed input."""
        
        try:
            # Handle different agent interfaces
            if hasattr(agent, 'process'):
                if agent_name == "spam_classifier":
                    user_personality = context.get("user_personality", "") if context else ""
                    return await agent.process(processed_input, user_personality)
                elif agent_name == "task_extractor":
                    user_personality = context.get("user_personality", "") if context else ""
                    return await agent.process(processed_input, user_personality)
                else:
                    return await agent.process(processed_input)
            else:
                # Handle agents with different method names
                if hasattr(agent, 'extract_relevant_email'):
                    return await agent.extract_relevant_email(processed_input, context.get("user_domain", ""))
                elif hasattr(agent, 'infer_domain'):
                    return await agent.infer_domain(processed_input)
                else:
                    return f"Agent {agent_name} has no standard process method"
                    
        except Exception as e:
            return {
                "error": f"Agent execution failed: {str(e)}",
                "agent": agent_name,
                "input": processed_input
            }

    async def batch_route(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Route multiple requests in batch.
        
        Args:
            requests: List of dicts with 'request' and optional 'context' keys
            
        Returns:
            List of routing results
        """
        results = []
        for req in requests:
            result = await self.route_request(
                req.get("request", ""),
                req.get("context", {})
            )
            results.append(result)
        return results