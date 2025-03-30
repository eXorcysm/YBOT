"""

This module instantiates the chatbot with the YBOT class.

"""

### Importing libraries ###

import logging
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama       import ChatOllama
from .agent                 import Agent
from .agent_test            import recall_memories
from .prompt                import build_prompt_template

### Environment settings ###

LLM            = "hf.co/backyardai/Psyonic-Cetacean-V1-20B-Ultra-Quality-GGUF:Q6_K"
# LLM            = "fimbul"
MAX_NEW_TOKENS = 256

logging.basicConfig(
    stream = sys.stdout, level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s"
)

### Class definition ###

class YBOT:
    """
    This class instantiates the chatbot.
    """

    def __init__(self, bot_id, user_id):
        self.ai   = bot_id
        self.user = user_id

        # Initialize chatbot agent.
        self.agent = Agent(user_id)

        # Initialize chatbot LLM.
        chatbot_llm = ChatOllama(
            model       = LLM,
            num_predict = MAX_NEW_TOKENS,
            temperature = 0.8
        )

        # Build prompt chain.
        prompt_template   = build_prompt_template(bot = bot_id, usr = user_id)
        prompt            = ChatPromptTemplate.from_template(prompt_template)
        self.prompt_chain = prompt | chatbot_llm

    def chat(self, query):
        """
        Send user query to chat engine and collect response.
        """

        agent_answer = self.agent.graph.stream(
            {"messages" : [("user", query)]}, config = self.agent.config
        )

        context = ""

        for chunk in agent_answer:
            context += recall_memories(chunk)

        if context:
            logging.info("[+] Recalled memories: [%s]", context)

        answer = self.prompt_chain.invoke({"context" : context, "query" : query})

        logging.info("[+] Model response: %s", answer.content)

        return answer.content
