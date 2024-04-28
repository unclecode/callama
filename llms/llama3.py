# llama3.py

import json, os
from jinja2 import Template
from typing import List, Dict, Union, Generator
from unsloth import FastLanguageModel
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread
from dotenv import load_dotenv
load_dotenv()


class CaLLama:
    def __init__(self, model_name: str, max_seq_length: int = 4096 * 2, dtype=None, load_in_4bit: bool = True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.text_streamer = None

        self.eos_token = '<|eot_id|>'
        self.lama3_template = Template(
            "{% if messages[0]['role'] == 'system' %}"\
        "<|start_header_id|>system<|end_header_id|>\n\n"\
        "{{ messages[0]['content'] }}\n"\
        "{{ tools }}\n"\
    "{% endif %}"\
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "<|start_header_id|>user<|end_header_id|>\n\n"\
            "{{ message['content'] }}\n"\
        "{% elif message['role'] == 'tool' %}"\
            "<|start_header_id|>assistant<|end_header_id|>\n\n"\
            "<functioncall> {{ message['content'] }}<|eot_id|>\n"\
        "{% elif message['role'] == 'tool_response' %}"\
            "<|start_header_id|>assistant<|end_header_id|>\n\n"\
            "{{ message['content'] }}\n"\
        "{% elif message['role'] == 'assistant' %}"\
            "<|start_header_id|>assistant<|end_header_id|>\n\n"\
            "{{ message['content'] }}<|eot_id|>\n"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if tool_call %}"\
        "<|start_header_id|>assistant<|end_header_id|>\n\n<functioncall> "\
    "{% endif %}"
        )

    def render(self, messages: List[Dict[str, str]], tools: List[Dict[str, str]], tool_call=False) -> str:
        # Ensure there is a system message at the beginning
        if messages[0]['role'] != 'system':
            messages.insert(0, {'role': 'system', 'content': ''})

        # Convert tools to a JSON string
        tools_json = []
        for tool in tools:
            tools_json.append(json.dumps(tool, indent=4))

        tools_json = '\n'.join(tools_json)

        # Render the template with the provided messages, tools, and add_generation_prompt
        rendered_string = self.lama3_template.render(messages=messages, tools=tools_json, tool_call=tool_call)

        return rendered_string

    def get_func_call(self, text: str, prompt: str) -> str:
        return text.split(prompt)[1].split(self.eos_token)[0]

    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            token=os.environ['HF_TOKEN']
        )
        FastLanguageModel.for_inference(self.model)
        self.text_streamer = TextStreamer(self.tokenizer)

    def completion(self, messages: List[Dict[str, str]], tools: List[Dict[str, str]], stream: bool = False,
                   max_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0) -> Union[str, Generator[str, None, None]]:
        prompt = self.render(messages, tools, tool_call=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        if stream:
            return self._stream_tokens(prompt, inputs, max_tokens, temperature, top_p)
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id,
                                          temperature=temperature, top_p=top_p)
            response = self.tokenizer.batch_decode(outputs)
            return self.get_func_call(response[0], prompt)

    def _stream_tokens(self, prompt, inputs, max_tokens, temperature, top_p) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, pad_token_id=self.tokenizer.eos_token_id)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        is_first = True
        for new_text in streamer:
            if not is_first and new_text.strip() != self.eos_token:
                yield new_text        
            is_first = False