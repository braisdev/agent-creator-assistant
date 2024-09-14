# LangGraph Simple Chatbot Template

[![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml)

This template demonstrates a simple chatbot implemented using [LangGraph](https://github.com/langchain-ai/langgraph), designed for [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio). The chatbot maintains persistent chat memory, allowing for coherent conversations across multiple interactions.

The core logic, defined in `src/agent/graph.py`, showcases a straightforward chatbot that responds to user queries while maintaining context from previous messages.

## What it does

The simple chatbot:

1. Takes a user **message** as input
2. Maintains a history of the conversation
3. Generates a response based on the current message and conversation history
4. Updates the conversation history with the new interaction

This template provides a foundation that can be easily customized and extended to create more complex conversational agents.

## Getting Started

Assuming you have already [installed LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download), to set up:

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Define required API keys in your `.env` file.

<!--
Setup instruction auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
-->

Set up your LLM API keys. This repo defaults to using [Claude](https://console.anthropic.com/login).

<!--
End setup instructions
-->

3. Customize the code as needed.
4. Open the folder in LangGraph Studio!

## How to customize

1. **Modify the system prompt**: The default system prompt is defined in [configuration.py](./src/agent/configuration.py). You can easily update this via configuration in the studio to change the chatbot's personality or behavior.
2. **Select a different model**: We default to Anthropic's Claude 3 Sonnet. You can select a compatible chat model using `provider/model-name` via configuration. Example: `openai/gpt-4-turbo-preview`.
3. **Extend the graph**: The core logic of the chatbot is defined in [graph.py](./src/agent/graph.py). You can modify this file to add new nodes, edges, or change the flow of the conversation.

You can also quickly extend this template by:

- Adding custom tools or functions to enhance the chatbot's capabilities.
- Implementing additional logic for handling specific types of user queries or tasks.
- Integrating external APIs or databases to provide more dynamic responses.

## Development

While iterating on your graph, you can edit past state and rerun your app from previous states to debug specific nodes. Local changes will be automatically applied via hot reload. Try experimenting with:

- Modifying the system prompt to give your chatbot a unique personality.
- Adding new nodes to the graph for more complex conversation flows.
- Implementing conditional logic to handle different types of user inputs.

Follow-up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

For more advanced features and examples, refer to the [LangGraph documentation](https://github.com/langchain-ai/langgraph). These resources can help you adapt this template for your specific use case and build more sophisticated conversational agents.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates, allowing you to analyze and optimize your chatbot's performance.

<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "agent": {
      "type": "object",
      "properties": {
        "system_prompt": {
          "type": "string",
          "default": "You are a helpful (if not sassy) personal assistant.\n\nSystem time: {system_time}"
        },
        "model_name": {
          "type": "string",
          "default": "anthropic/claude-3-5-sonnet-20240620",
          "environment": [
            {
              "value": "anthropic/claude-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.0",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.1",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-5-sonnet-20240620",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-haiku-20240307",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-opus-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-sonnet-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-instant-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "fireworks/gemma2-9b-it",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-70b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-70b-instruct-hf",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-8b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-8b-instruct-hf",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-405b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-405b-instruct-long",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-70b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-8b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mixtral-8x22b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mixtral-8x7b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mixtral-8x7b-instruct-hf",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mythomax-l2-13b",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/phi-3-vision-128k-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/phi-3p5-vision-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/starcoder-16b",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/yi-large",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      }
    }
  }
}
-->
