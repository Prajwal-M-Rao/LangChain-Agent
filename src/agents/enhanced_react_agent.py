"""
Enhanced ReAct Agent for Wikipedia and web search tasks.
Implements reasoning and acting pattern with tool selection and memory.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..tools.base_tool import BaseTool
from ..tools.wikipedia_tool import WikipediaTool
from ..tools.duckduckgo_tool import DuckDuckGoTool
from ..tools.wolfram_alpha_tool import WolframAlphaTool
from ..memory.conversation_memory import ConversationMemory
from ..memory.response_cache import ResponseCache
from ..llm.local_llm import LocalLLM
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector

logger = get_logger(__name__)


class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"


@dataclass
class AgentStep:
    state: AgentState
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    
    
class EnhancedReActAgent:
    """Enhanced ReAct agent with memory, caching, and advanced reasoning."""
    
    def __init__(
        self,
        llm: LocalLLM,
        max_iterations: int = 10,
        memory_size: int = 100,
        use_cache: bool = True,
        temperature: float = 0.1
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Initialize tools
        self.tools = self._initialize_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Initialize memory and cache
        self.memory = ConversationMemory(max_size=memory_size)
        self.cache = ResponseCache() if use_cache else None
        
        # Initialize metrics
        self.metrics = MetricsCollector()
        
        # ReAct pattern templates
        self.system_prompt = self._create_system_prompt()
        
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize available tools."""
        tools = []
        
        try:
            tools.append(WikipediaTool())
            tools.append(DuckDuckGoTool())
            tools.append(WolframAlphaTool())
        except Exception as e:
            logger.warning(f"Failed to initialize some tools: {e}")
            
        return tools
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for ReAct reasoning."""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
        
        return f"""You are an enhanced research assistant that uses the ReAct (Reasoning and Acting) pattern.
        
Available tools:
{tool_descriptions}

Instructions:
1. Think step by step about the user's question
2. Use tools when you need information
3. Provide comprehensive, well-structured answers
4. Always cite your sources when using tool results

Format your response as:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [result from the tool]

When you have enough information to answer, format your final response as:
Thought: I have enough information to provide a comprehensive answer.
Final Answer: [Your complete answer with proper citations]

Remember to:
- Use tools efficiently to gather relevant information
- Cross-reference information from multiple sources when possible
- Be precise and factual in your responses
- Always provide sources for factual claims
"""

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the enhanced ReAct agent on a query."""
        logger.info(f"Starting ReAct agent for query: {query}")
        
        # Check cache first
        if self.cache:
            cached_result = await self.cache.get(query)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
        
        # Initialize tracking
        steps = []
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Add query to memory
            self.memory.add_message("user", query)
            
            # Get conversation context
            conversation_context = self.memory.get_recent_context(n=5)
            
            # Run ReAct loop
            result = await self._react_loop(query, conversation_context, steps)
            
            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            self.metrics.record_query(
                query=query,
                response_time=end_time - start_time,
                steps_taken=len(steps),
                success=result.get("success", False)
            )
            
            # Cache successful results
            if self.cache and result.get("success", False):
                await self.cache.set(query, result)
            
            # Add response to memory
            self.memory.add_message("assistant", result.get("answer", ""))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": steps
            }
    
    async def _react_loop(
        self, 
        query: str, 
        context: List[Dict[str, str]], 
        steps: List[AgentStep]
    ) -> Dict[str, Any]:
        """Execute the ReAct reasoning loop."""
        
        # Prepare initial prompt
        context_str = self._format_context(context)
        prompt = f"{self.system_prompt}\n\nContext:\n{context_str}\n\nUser Query: {query}\n\nLet's work through this step by step."
        
        conversation_history = prompt
        
        for iteration in range(self.max_iterations):
            logger.debug(f"ReAct iteration {iteration + 1}")
            
            # Get LLM response
            response = await self.llm.generate(
                conversation_history,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            # Parse response
            step = self._parse_response(response)
            steps.append(step)
            
            if step.state == AgentState.FINISHED:
                return {
                    "success": True,
                    "answer": step.observation or step.thought,
                    "steps": steps,
                    "iterations": iteration + 1
                }
            
            # Execute action if needed
            if step.action and step.action_input:
                observation = await self._execute_action(step.action, step.action_input)
                step.observation = observation
                
                # Add to conversation history
                conversation_history += f"\n\nThought: {step.thought}"
                if step.action:
                    conversation_history += f"\nAction: {step.action}"
                    conversation_history += f"\nAction Input: {step.action_input}"
                    conversation_history += f"\nObservation: {observation}"
            else:
                conversation_history += f"\n\nThought: {step.thought}"
        
        return {
            "success": False,
            "error": "Maximum iterations reached",
            "steps": steps,
            "iterations": self.max_iterations
        }
    
    def _parse_response(self, response: str) -> AgentStep:
        """Parse LLM response into an AgentStep."""
        
        # Look for final answer
        if "Final Answer:" in response:
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)
            if final_answer_match:
                return AgentStep(
                    state=AgentState.FINISHED,
                    thought="Providing final answer",
                    observation=final_answer_match.group(1).strip()
                )
        
        # Parse thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer)|\Z)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else response.strip()
        
        # Parse action
        action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", response)
        action = action_match.group(1).strip() if action_match else None
        
        # Parse action input
        action_input_match = re.search(r"Action Input:\s*(.*?)(?=\n|$)", response, re.DOTALL)
        action_input = action_input_match.group(1).strip() if action_input_match else None
        
        # Determine state
        if action and action_input:
            state = AgentState.ACTING
        else:
            state = AgentState.THINKING
        
        return AgentStep(
            state=state,
            thought=thought,
            action=action,
            action_input=action_input
        )
    
    async def _execute_action(self, action: str, action_input: str) -> str:
        """Execute a tool action."""
        try:
            tool = self.tool_map.get(action)
            if not tool:
                available_tools = ", ".join(self.tool_map.keys())
                return f"Error: Unknown tool '{action}'. Available tools: {available_tools}"
            
            result = await tool.execute(action_input)
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return f"Error executing {action}: {str(e)}"
    
    def _format_context(self, context: List[Dict[str, str]]) -> str:
        """Format conversation context for prompt."""
        if not context:
            return "No previous context."
        
        formatted = []
        for msg in context:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.metrics.get_summary()
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Response cache cleared")