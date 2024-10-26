# This script is used to generate wide range of prompts
# required pkg: langchain_core
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
    convert_to_openai_function,
)
from typing import Literal
import json


class SystemPrompt:
    # Static system prompts:
    practical = """Never use "As an AI Language Model" when answering questions.
Keep the responses brief and informative, avoid superfluous language and unnecessarily long explanations.
If you don't know, say that you don't know.
Your answers should be on point, succinct and useful. Each response should be written with maximum usefulness in mind rather than being polite.
"""
    solve_problem = f"""{practical}
If you think you can't do something, don't put the burden on the user to do it, instead try to exhaust all of your options first.
When solving problems, take a breath and do it step by step.
"""

    def __init__(self):
        pass

    def one_tool_calling(self, tools):
        """
        TODO: system prompt for tool calling
        param tools: list of tools, each tool is a python function with descriptions
        """
        assert len(tools) > 0, "Need to provide at least one function in tools"
        assert all(
            callable(func) for func in tools
        ), "All elements in tools must be callable"
        tool_string = list(
            map(lambda x: json.dumps(convert_to_openai_function(x)), tools)
        )
        prompt = """
You are a helpful assistant that takes a question and finds the most appropriate tool to execute, 
along with the parameters required to run the tool.

Always respond with a single JSON object containing exactly two keys:
    name: str = Field(description="Name of the function to run (null if you don't need to invoke a tool)")
    args: dict = Field(description="Arguments for the function call (empty array if you don't need to invoke a tool or if no arguments are needed for the tool call)")

Don't start your answers with  "```json" or "Here is the JSON response", just give the JSON.

The tools you have access to are: 
{tool_string}
        """
        return prompt.format(tool_string=tool_string)

    def multiple_tool_calling(self, tools):
        pass

    def summarization(self):
        pass
