# CaLLama: Democratizing Function Calling Capabilities for Open-Source Language Models

This repository is dedicated to advancing the "function-call" features for open-source large language models (LLMs). We believe that the future of AI, specifically AI agents, depends on proper function-calling capabilities. While proprietary models like OpenAI's have these features, it is crucial for the open-source community to have access to high-quality function-calling abilities to democratize AI.

Recently, Facebook released LLaMA3, perhaps the best open-source LLM available. We have fine-tuned and created a version of LLaMA3 that natively supports function calls.

## Usage

To use the models in this repository, follow these steps:

1. Clone the repository:
```
git clone https://github.com/unclecode/fllm.git
```

2. Create a virtual environment and activate it:
```
conda create --name env python=3.10
conda activate env
```

3. Install PyTorch if you haven't already:
```
conda install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
```

4. Install the required dependencies:
```
python setup.py
```
5. Add your HuggingFace token in ".token.env", and then rename the file to ".env".

6. Run the example code in the `examples` folder to see the models in action.

You can also refer to the `callama.py` file in the `llms` folder to see the LLM chat template.

For a more interactive experience, check out the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://tinyurl.com/ucfllm)

## Features TODO List

[x] Single function detection
[x] Support for various model sizes and quantization levels
[x] Available as a LoRA adapter that can be merged with many models
[ ] Multi-function detection
[ ] Function binding, allowing the model to detect the order of execution and bind the output of one function to another
[ ] Fine-tuning models with less than 1B parameters for efficient function calling

## Models

The following models are available on Hugging Face:

- [unclecode/llama3-function-call-lora-adapter-240424](https://huggingface.co/unclecode/llama3-function-call-lora-adapter-240424)
- [unclecode/llama3-function-call-16bit-240424](https://huggingface.co/unclecode/llama3-function-call-16bit-240424)
- [unclecode/llama3-function-call-4bit-240424](https://huggingface.co/unclecode/llama3-function-call-4bit-240424)
- [unclecode/llama3-function-call-Q4_K_M_GGFU-240424](https://huggingface.co/unclecode/llama3-function-call-Q4_K_M_GGFU-240424)
- [unclecode/tinyllama-function-call-lora-adapter-250424](https://huggingface.co/unclecode/tinyllama-function-call-lora-adapter-250424)
- [unclecode/tinyllama-function-call-16bit-250424](https://huggingface.co/unclecode/tinyllama-function-call-16bit-250424)
- [unclecode/tinyllama-function-call-Q4_K_M_GGFU-250424](https://huggingface.co/unclecode/tinyllama-function-call-Q4_K_M_GGFU-250424)

## Dataset

The models were fine-tuned using a modified version of the `ilacai/glaive-function-calling-v2-sharegpt` dataset, which can be found at [unclecode/glaive-function-calling-llama3](https://huggingface.co/datasets/unclecode/glaive-function-calling-llama3).

## Contributing

We welcome contributions from the community. If you are interested in joining this project or have any questions, please open an issue in this repository.

Twitter (X): https://x.com/unclecode

## License

These models are released under the Apache 2.0 license.
