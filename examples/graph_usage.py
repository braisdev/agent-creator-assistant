# usage.py

import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent.graph import graph

load_dotenv()


def main():

    # 2) Now we can do ainvoke on the graph
    config = {"configurable": {"thread_id": "1", "tenant_id": "brais", "expert_id": "1"}}

    input_message = HumanMessage(content="Hello! the expert name is Brais")
    initial_state = {"messages": [input_message]}

    result = graph.invoke(initial_state, config)
    print("Result:", result)


if __name__ == "__main__":
    main()
