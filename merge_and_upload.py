#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge LoRA weights with base model and upload to HuggingFace
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure these paths before running
os.environ["http_proxy"] = ""  # Set if needed
os.environ["https_proxy"] = ""  # Set if needed
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

BASE_MODEL_PATH = "/path/to/Qwen3-8B"
LORA_PATH = "/path/to/checkpoints/checkpoint_step_xxx"
MERGED_MODEL_PATH = "/tmp/merged-model"
HF_REPO_NAME = "your-username/your-model-name"

def main():
    print("=" * 60)
    print("Step 1: Loading base model...")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
    )

    print(f"Base model loaded: {BASE_MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Step 2: Loading LoRA adapter...")
    print("=" * 60)

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.bfloat16,
    )

    print(f"LoRA adapter loaded: {LORA_PATH}")

    print("\n" + "=" * 60)
    print("Step 3: Merging LoRA weights...")
    print("=" * 60)

    merged_model = model.merge_and_unload()

    print("LoRA weights merged successfully!")

    print("\n" + "=" * 60)
    print("Step 4: Saving merged model...")
    print("=" * 60)

    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print(f"Merged model saved to: {MERGED_MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Step 5: Uploading to HuggingFace...")
    print("=" * 60)

    from huggingface_hub import HfApi, login

    login(token=os.environ["HF_TOKEN"])

    api = HfApi()

    try:
        api.create_repo(
            repo_id=HF_REPO_NAME,
            repo_type="model",
            exist_ok=True,
        )
        print(f"Repository created/exists: {HF_REPO_NAME}")
    except Exception as e:
        print(f"Note: {e}")

    api.upload_folder(
        folder_path=MERGED_MODEL_PATH,
        repo_id=HF_REPO_NAME,
        repo_type="model",
    )

    print(f"\nModel uploaded successfully!")
    print(f"URL: https://huggingface.co/{HF_REPO_NAME}")

if __name__ == "__main__":
    main()
