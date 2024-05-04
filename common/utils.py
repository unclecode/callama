import json
from jinja2 import Template
from typing import List, Dict
import os

def render(messages: List[Dict[str, str]], tools: List[Dict[str, str]], tool_call=False) -> str:
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the template from the "prompt.jinja2" file in the same directory
    with open(os.path.join(current_dir, "prompt.jinja2"), "r") as template_file:
        template_str = template_file.read()
    
    # Create a Jinja2 template from the loaded template string
    template = Template(template_str)
    
    # Ensure there is a system message at the beginning
    if messages[0]['role'] != 'system':
        messages.insert(0, {'role': 'system', 'content': ''})

    # Convert tools to a JSON string
    tools_json = []
    for tool in tools:
        tools_json.append(json.dumps(tool, indent=4))

    tools_json = '\n'.join(tools_json)

    # Render the template with the provided messages, tools, and add_generation_prompt
    rendered_string = template.render(messages=messages, tools=tools_json, tool_call=tool_call)

    return rendered_string

def extract_arguments(text: str, prompt: str) -> str:
    return text.split(prompt)[1].split(self.eos_token)[0]