from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver # Recordar conversación sólo en memoria	
from langgraph.checkpoint.sqlite import SqliteSaver # Recordar conversación en una base de datos SQLite
import sqlite3
import os

load_dotenv()

# memoria

config = {"configurable": {"thread_id": "1"}}

# checkpointer = MemorySaver()
checkpointer = SqliteSaver(conn=sqlite3.connect("chat.db", check_same_thread=False))

# tools

tool = TavilySearch(max_results=3)

# llm

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

llm_with_tools = llm.bind_tools([tool])

# grafo

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=[tool])

graph_builder = StateGraph(State)
graph_builder.add_edge(START, "agent")
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile(checkpointer=checkpointer)

# ejecución

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()