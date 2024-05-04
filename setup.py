import subprocess
import sys
import torch

def install(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", package])

def main():
    print("Retrieving CUDA device capability...")
    major_version, minor_version = torch.cuda.get_device_capability()

    print("Installing unsloth package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"])

    if major_version >= 8:
        print("Installing packages for newer NVIDIA GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)...")
        install(["packaging", "ninja", "einops", "flash-attn", "xformers", "trl", "peft", "accelerate", "bitsandbytes", "pydantic", "jinja2", "python-dotenv", "ollama"])
    else:
        print("Installing packages for older NVIDIA GPUs (V100, Tesla T4, RTX 20xx)...")
        install(["xformers", "trl", "peft", "accelerate", "bitsandbytes", "pydantic", "jinja2", "python-dotenv", "ollama"])

    print("Installation complete.")

if __name__ == "__main__":
    main()
