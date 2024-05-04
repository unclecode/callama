# llama3.py

import json, os
from jinja2 import Template
from typing import List, Dict, Union, Generator
from unsloth import FastLanguageModel
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread
from dotenv import load_dotenv
load_dotenv()
from common.utils import render, extract_arguments

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
        prompt = render(messages, tools, tool_call=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        if stream:
            return self._stream_tokens(prompt, inputs, max_tokens, temperature, top_p)
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id,
                                          temperature=temperature, top_p=top_p)
            response = self.tokenizer.batch_decode(outputs)
            return extract_arguments(response[0], prompt)

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