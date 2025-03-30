"""

This module manages the chatbot agent and its tools and functions.

"""

### Importing libraries ###

from typing import List

import sys

from IPython.display             import Image
from IPython.display             import display
from langchain_core.messages     import get_buffer_string
from langchain_core.prompts      import ChatPromptTemplate
from langchain_core.runnables    import RunnableConfig
from langchain_ollama            import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph             import END, START
from langgraph.graph             import MessagesState
from langgraph.graph             import StateGraph
from langgraph.prebuilt          import ToolNode
from transformers                import AutoTokenizer
from .agent_test                 import run_tests
from .agent_tools                import save_recall_memory
from .agent_tools                import search_recall_memories
from .agent_tools                import search_web

### Environment settings ###

sys.path.append("../")

AGENT_MODEL = "qwen2.5:14b"
PROMPT      = "./data/prompts/agent.txt"
TOKEN_MODEL = "Qwen/Qwen2.5-14B"

### Class definitions ###

class State(MessagesState):
    """
    This object is used to store memories to be retrieved based on conversation context.
    """

    recall_memories: List[str]

class Agent():
    """
    This class instantiates the AI agent.
    """

    def __init__(self, user_id):
        self.config    = {"configurable" : {"user_id" : user_id, "thread_id" : "1"}}
        self.llm       = ChatOllama(model = AGENT_MODEL, temperature = 0)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKEN_MODEL)
        self.tools     = [save_recall_memory, search_recall_memories, search_web]

        self.graph          = self.build_graph()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.prompt         = self.build_prompt()
        self.user           = user_id

    def build_graph(self):
        """
        Build agent graph.
        """

        # Initialize graph.
        graph = StateGraph(State)

        # Add nodes.
        graph.add_node(self.load_memories)
        graph.add_node(self.query_agent)
        graph.add_node("tools", ToolNode(self.tools))

        # Add edges.
        graph.add_edge(START, "load_memories")
        graph.add_edge("load_memories", "query_agent")
        graph.add_conditional_edges("query_agent", self.route_tools, ["tools", END])
        graph.add_edge("tools", "query_agent")

        # Compile graph.
        graph_memory = MemorySaver()
        lang_graph   = graph.compile(checkpointer = graph_memory)

        return lang_graph

    def build_prompt(self):
        """
        Build prompt for AI agent.
        """

        with open(PROMPT, encoding = "utf-8") as txt:
            agent_prompt = txt.read()

        prompt = ChatPromptTemplate.from_messages(
            [("placeholder", "{messages}"), ("system", agent_prompt)]
        )

        return prompt

    def load_memories(self, state: State, config: RunnableConfig) -> State:
        """
        Load memories for current conversation.
        """

        query  = get_buffer_string(state["messages"])
        query  = self.tokenizer.decode(self.tokenizer.encode(query)[:2048])
        recall = search_recall_memories.invoke(query, config)

        return {
            "recall_memories" : recall
        }

    def query_agent(self, state: State) -> State:
        """
        Process current state and generate response using agent model.
        """

        prompt_chain = self.prompt | self.llm_with_tools

        recall = (
            "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
        )

        response = prompt_chain.invoke(
            {
                "messages"        : state["messages"],
                "recall_memories" : recall,
            }
        )

        return {
            "messages" : [response]
        }

    def route_tools(self, state: State):
        """
        Determine whether to use tools or end conversation based on last message.
        """

        msg = state["messages"][-1]

        if msg.tool_calls:
            return "tools"

        return END

    def show_graph(self):
        """
        Show graph architecture.
        """

        display(Image(self.graph.get_graph().draw_mermaid_png()))

def main():
    """
    Initialize AI agent and run tests.
    """

    agent = Agent()

    run_tests(agent.graph)

if __name__ == "__main__":
    main()
