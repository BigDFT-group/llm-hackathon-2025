import os

import asyncio
import getpass

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.human import HumanMessage

from MainAgent.utils import remote_run_code
from OntoFlow.agent.Onto_wa_rag.retriever_adapter import retriever_tool
from OntoFlow.agent.Onto_wa_rag.provider.get_key import get_openai_key
from OntoFlow.agent.Onto_wa_rag.CONSTANT import API_KEY_PATH


SYSTEM_PROMPT = """You are a helpful AI assistant that helps people find information and execute Python functions remotely. You may execute functions using tools when asked."""


class Chat:
    def __init__(self, model: str = "gpt-5", system_prompt: str = SYSTEM_PROMPT):
        if not "OPENAI_API_KEY" in os.environ:
            openai_key = get_openai_key(api_key_path=API_KEY_PATH)
            if openai_key == "":
                openai_key = getpass.getpass("Please enter the OpenAI API key:")
                os.environ["OPENAI_API_KEY"] = openai_key
        
        # create the agent
        self.model = ChatOpenAI(model=model, temperature=0)
        self.agent = create_react_agent(
            model=self.model,
            tools=[retriever_tool, remote_run_code],
            prompt=system_prompt,
            name="remotemanager_agent",
        )

        self._history = None

    async def chat(self, prompt: str):
        # Create the chat format
        msg = {"role": "user", "content": prompt}
        # if the history exists (we're responding), take from there
        content = {"messages": []}
        if self._history is not None:
            content = self._history.copy()
        # add the latest message
        content["messages"].append(msg)
        # invoke the agent and update the history
        self._history = await self.agent.ainvoke(content)

        return self._history.get("messages", [])[-1].content

    def print_history(self):
        if self._history is None:
            return "No Chat History"
        
        for i, message in enumerate(self._history["messages"]):

            source = "User" if isinstance(message, HumanMessage) else "Bot"
            
            print(f"\n\n#### Message {i+1} ####")
            print(f"{source}:")
            print(message.content)

    @property
    def history(self):
        return self.print_history()
