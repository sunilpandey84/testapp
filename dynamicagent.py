import json
import importlib
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass
from pathlib import Path

import langchain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import Annotated, TypedDict

def web_search():
    pass

from dotenv import load_dotenv

load_dotenv()

# State definition for the workflow
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_agent: str
    context: Dict[str, Any]
    memory: Dict[str, Any]


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    name: str
    role: str
    llm_config: Dict[str, Any]
    prompt: str
    tools: List[Dict[str, Any]]
    memory_config: Dict[str, Any]
    routes: List[str]


class LLMFactory:
    """Factory for creating different types of LLMs"""

    @staticmethod
    def create_llm(llm_config: Dict[str, Any]):
        """Create an LLM instance based on configuration"""
        llm_type = llm_config.get("type", "").lower()
        model = llm_config.get("model", "")
        temperature = llm_config.get("temperature", 0.7)
        max_tokens = llm_config.get("max_tokens", 1000)

        try:
            if llm_type == "openai":
                if model.startswith("gemini"):
                    return ChatGoogleGenerativeAI(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **{k: v for k, v in llm_config.items()
                           if k not in ["type", "model", "temperature", "max_tokens"]}
                    )
                else:
                    return ChatGoogleGenerativeAI(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **{k: v for k, v in llm_config.items()
                           if k not in ["type", "model", "temperature", "max_tokens"]}
                    )

            elif llm_type == "anthropic":
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **{k: v for k, v in llm_config.items()
                       if k not in ["type", "model", "temperature", "max_tokens"]}
                )

            elif llm_type == "local":
                # For local models, you might use Ollama or other local LLM wrappers
                from langchain.llms import Ollama
                return Ollama(
                    model=model,
                    temperature=temperature,
                    **{k: v for k, v in llm_config.items()
                       if k not in ["type", "model", "temperature"]}
                )

            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")

        except ImportError as e:
            raise ImportError(f"Required package for {llm_type} not installed: {e}")
        except Exception as e:
            raise ValueError(f"Error creating LLM {llm_type}: {e}")


class ToolFactory:
    """Factory for loading tools dynamically"""

    @staticmethod
    def load_tool(tool_config: Dict[str, Any]) -> BaseTool:
        """Load a tool based on configuration"""
        tool_type = tool_config.get("type", "")

        try:
            if tool_type == "python_function":
                return ToolFactory._load_python_function(tool_config)
            elif tool_type == "langchain_tool":
                return ToolFactory._load_langchain_tool(tool_config)
            else:
                raise ValueError(f"Unsupported tool type: {tool_type}")

        except Exception as e:
            raise ValueError(f"Error loading tool {tool_config.get('name', 'unknown')}: {e}")

    @staticmethod
    def _load_python_function(tool_config: Dict[str, Any]) -> BaseTool:
        """Load a Python function as a tool"""
        from langchain.tools import Tool

        path = tool_config["path"]
        name = tool_config["name"]
        description = tool_config.get("description", f"Tool: {name}")

        # Parse module path and function name
        module_path, func_name = path.rsplit(".", 1)

        # Import the module and function
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)

        return Tool(
            name=name,
            description=description,
            func=func
        )

    @staticmethod
    def _load_langchain_tool(tool_config: Dict[str, Any]) -> BaseTool:
        """Load a LangChain tool"""
        path = tool_config["path"]

        # Parse module path and class name
        module_path, class_name = path.rsplit(".", 1)

        # Import the module and class
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)

        # Initialize with parameters
        params = tool_config.get("params", {})
        return tool_class(**params)


class MemoryFactory:
    """Factory for creating different types of memory"""

    @staticmethod
    def create_memory(memory_config: Dict[str, Any]):
        """Create a memory instance based on configuration"""
        memory_type = memory_config.get("type", "ConversationBufferMemory")
        params = memory_config.get("params", {})

        try:
            if memory_type == "ConversationBufferMemory":
                return ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    **params
                )
            elif memory_type == "ConversationSummaryMemory":
                return ConversationSummaryMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    **params
                )
            else:
                raise ValueError(f"Unsupported memory type: {memory_type}")

        except Exception as e:
            raise ValueError(f"Error creating memory {memory_type}: {e}")


class AgentNode:
    """Represents a single agent node in the workflow"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.llm = LLMFactory.create_llm(config.llm_config)
        self.tools = [ToolFactory.load_tool(tool_config) for tool_config in config.tools]
        self.memory = MemoryFactory.create_memory(config.memory_config)
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a prompt template for the agent"""
        return ChatPromptTemplate.from_messages([
            ("system", f"You are {self.config.role}. {self.config.prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def __call__(self, state: AgentState) -> AgentState:
        """Execute the agent"""
        try:
            # Get the latest message
            if not state["messages"]:
                return state

            latest_message = state["messages"][-1]
            input_text = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)

            # Create agent executor if tools are available
            if self.tools:
                from langchain.agents import create_react_agent, AgentExecutor

                agent = create_react_agent(self.llm, self.tools, self.prompt_template)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    memory=self.memory,
                    verbose=True,
                    max_iterations=3,
                    handle_parsing_errors=True
                )

                result = agent_executor.invoke({
                    "input": input_text,
                    "chat_history": state.get("memory", {}).get(self.name, [])
                })

                response_content = result.get("output", "No response generated")
            else:
                # Simple LLM call without tools
                messages = [
                    {"role": "system", "content": f"You are {self.config.role}. {self.config.prompt}"},
                    {"role": "user", "content": input_text}
                ]

                response = self.llm.invoke(messages)
                response_content = response.content if hasattr(response, 'content') else str(response)

            # Update state
            new_state = state.copy()
            new_state["messages"].append(AIMessage(content=response_content))
            new_state["current_agent"] = self.name

            # Update memory
            if "memory" not in new_state:
                new_state["memory"] = {}
            if self.name not in new_state["memory"]:
                new_state["memory"][self.name] = []

            new_state["memory"][self.name].extend([
                HumanMessage(content=input_text),
                AIMessage(content=response_content)
            ])

            return new_state

        except Exception as e:
            error_msg = f"Error in agent {self.name}: {str(e)}"
            print(error_msg)

            new_state = state.copy()
            new_state["messages"].append(AIMessage(content=error_msg))
            new_state["current_agent"] = self.name
            return new_state


class WorkflowRouter:
    """Handles routing between agents"""

    def __init__(self, agent_configs: List[AgentConfig]):
        self.routes = {}
        for config in agent_configs:
            self.routes[config.name] = config.routes

    def route(self, state: AgentState) -> str:
        """Determine the next agent based on current state"""
        current_agent = state.get("current_agent", "")

        if current_agent in self.routes:
            routes = self.routes[current_agent]
            if routes:
                # Simple routing - take the first route
                # You can implement more sophisticated routing logic here
                return routes[0]

        return END


class CodelessWorkflowBuilder:
    """Main builder class for creating workflows from JSON templates"""

    def __init__(self, template_path: str):
        self.template_path = template_path
        self.config = self._load_template()
        self.agent_configs = self._parse_agent_configs()
        self.agents = {}
        self.router = WorkflowRouter(self.agent_configs)

    def _load_template(self) -> Dict[str, Any]:
        """Load JSON template from file"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file: {e}")

    def _parse_agent_configs(self) -> List[AgentConfig]:
        """Parse agent configurations from template"""
        agents_data = self.config.get("agents", [])
        configs = []

        for agent_data in agents_data:
            try:
                config = AgentConfig(
                    name=agent_data["name"],
                    role=agent_data["role"],
                    llm_config=agent_data["llm"],
                    prompt=agent_data["prompt"],
                    tools=agent_data.get("tools", []),
                    memory_config=agent_data.get("memory", {"type": "ConversationBufferMemory"}),
                    routes=agent_data.get("routes", [])
                )
                configs.append(config)
            except KeyError as e:
                raise ValueError(f"Missing required field in agent config: {e}")

        return configs

    def build_workflow(self) -> StateGraph:
        """Build the complete LangGraph workflow"""
        # Create state graph
        workflow = StateGraph(AgentState)

        # Create agent nodes
        for config in self.agent_configs:
            try:
                agent_node = AgentNode(config)
                self.agents[config.name] = agent_node
                workflow.add_node(config.name, agent_node)
            except Exception as e:
                print(f"Warning: Failed to create agent {config.name}: {e}")

        # Set entry point (first agent in the list)
        if self.agent_configs:
            workflow.set_entry_point(self.agent_configs[0].name)

        # Add edges based on routes
        for config in self.agent_configs:
            for route in config.routes:
                if route == "END":
                    workflow.add_edge(config.name, END)
                elif route in self.agents:
                    workflow.add_edge(config.name, route)
                else:
                    print(f"Warning: Route {route} from {config.name} not found")

        # Add conditional edges for dynamic routing
        for agent_name in self.agents:
            workflow.add_conditional_edges(
                agent_name,
                self.router.route,
                {route: route for routes in self.router.routes.values() for route in routes if route != "END"}
            )

        return workflow

    def compile_workflow(self, checkpointer=None):
        """Compile the workflow for execution"""
        workflow = self.build_workflow()

        if checkpointer is None:
            # Use SQLite checkpointer for memory
            checkpointer = SqliteSaver.from_conn_string(":memory:")

        return workflow.compile(checkpointer=checkpointer)


# Example usage and demonstration
def create_example_template():
    """Create an example JSON template"""
    template = {
        "agents": [
            {
                "name": "researcher",
                "role": "Research Assistant",
                "llm": {
                    "type": "openai",
                    "model": "gemini-1.5-flash",
                    "temperature": 0.2
                },
                "prompt": "You are a thorough researcher. Analyze questions and provide well-researched answers.",
                "tools": [],
                "memory": {
                    "type": "ConversationBufferMemory",
                    "params": {}
                },
                "routes": ["summarizer"]
            },
            {
                "name": "summarizer",
                "role": "Content Summarizer",
                "llm": {
                    "type": "openai",
                    "model": "gemini-1.5-flash",
                    "temperature": 0.1
                },
                "prompt": "You are an expert at creating concise, accurate summaries.",
                "tools": [],
                "memory": {
                    "type": "ConversationSummaryMemory",
                    "params": {}
                },
                "routes": ["END"]
            }
        ]
    }

    with open("example_workflow.json", "w") as f:
        json.dump(template, f, indent=2)

    return "example_workflow.json"


def main():
    """Example of how to use the workflow builder"""

    # Create example template
    template_file = create_example_template()

    try:
        # Build workflow from template
        builder = CodelessWorkflowBuilder(template_file)
        compiled_workflow = builder.compile_workflow()

        # Example invocation
        initial_state = {
            "messages": [HumanMessage(content="What is artificial intelligence?")],
            "current_agent": "",
            "context": {},
            "memory": {}
        }

        # Execute workflow
        print("Starting workflow execution...")
        result = compiled_workflow.invoke(initial_state)

        print("\nWorkflow completed!")
        print("Final messages:")
        for msg in result["messages"]:
            print(f"- {msg.content[:100]}...")

        # Stream execution example
        print("\n" + "=" * 50)
        print("Streaming execution example:")

        stream_state = {
            "messages": [HumanMessage(content="Explain machine learning briefly")],
            "current_agent": "",
            "context": {},
            "memory": {}
        }

        for chunk in compiled_workflow.stream(stream_state):
            print(f"Chunk: {chunk}")

    except Exception as e:
        print(f"Error: {e}")

    # Clean up
    import os
    if os.path.exists(template_file):
        os.remove(template_file)


if __name__ == "__main__":
    main()
