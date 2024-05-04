import ollama
from common.utils import render, extract_arguments

def use_with_helpwr():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to the following functions. Use them if required -",
        },
        {
            "role": "user",
            "content": "Hi, send an email to tom@kidocode.com and ask him to join our weekend party?",
        },
    ]

    tools = [
        {
            "name": "send_email",
            "description": "Send an email for the given recipient and message",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "The email address of the recipient",
                    },
                    "message": {"type": "string", "description": "The message to send"},
                },
                "required": ["recipient", "message"],
            },
        }
    ]

    rendered_string = render(messages, tools, tool_call=True)

    ollama = ollama.chat(
        model="unclecode/tinycallama",
        messages=[
            {
                "role": "user",
                "content": rendered_string,
            },
        ],
    )
    print(ollama["message"]["content"])


def use_with_prompt():
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant with access to the following functions. Use them if required -
    {
    "name": "send_email",
    "description": "Send an email for the given recipient and message",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "The email address of the recipient"
            },
            "message": {
                "type": "string",
                "description": "The message to send"
            }
        },
        "required": [
            "recipient",
            "message"
        ]
    }
    }
    <|start_header_id|>user<|end_header_id|>

    Hi, send an email to tom@kidocode.com and ask him to join our weekend party?
    <|start_header_id|>assistant<|end_header_id|>

    <functioncall> """

    response = ollama.chat(
        model="unclecode/tinycallama",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    print("Using Ollama with helper")
    use_with_helper()
    
    print("Using Ollama with prompt")
    use_with_prompt()