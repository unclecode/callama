# test_examples.py
try:
    from llms.llama3 import CaLLama
except ImportError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from llms.llama3 import CaLLama

# Test messages
arithmetic_messages = [
    {'role': 'user', 'content': 'Calculate the 3 * 12 + 3?'},
    {'role': 'tool', 'content': '{"name": "mul", "arguments": \'{"a": 3, "b": 12}\'}  '},
    {'role': 'tool_response', 'content': '{"result": 36}'},
]

email_messages = [
    {"role": "system", "content": "You are a helpful assistant with access to the following functions. Use them if required -"},
    {'role': 'user', 'content': "Hi, send an email to tom@kidocode.com and ask him to join our weekend party?"},
]

arithmetic_tools = [
    {
        "name": "add",
        "description": "Calculate the sum of two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number to add"
                },
                "b": {
                    "type": "integer",
                    "description": "The second number to add"
                }
            },
            "required": [
                "a",
                "b"
            ]
        }
    },
    {
        "name": "mul",
        "description": "Calculate the product of two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number to multiply"
                },
                "b": {
                    "type": "integer",
                    "description": "The second number to multiply"
                }
            },
            "required": [
                "a",
                "b"
            ]
        }
    }
]

email_tools = [
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
]

if __name__ == "__main__":
    # Current models
    llama_model_name="unclecode/llama3-function-call-lora-adapter-240424"
    tinyllama_model_name="unclecode/tinyllama-function-call-lora-adapter-250424"

    # Usage
    llama3 = CaLLama(model_name=llama_model_name)
    llama3.load_model()

    # Non-streaming completion for arithmetic
    arithmetic_result = llama3.completion(arithmetic_messages, arithmetic_tools, stream=False)
    print("Arithmetic Result:")
    print(arithmetic_result)

    # Streaming completion for email
    print("\nEmail Result (Streaming):")
    for token in llama3.completion(email_messages, email_tools, stream=True):
        print(token, end="", flush=True)