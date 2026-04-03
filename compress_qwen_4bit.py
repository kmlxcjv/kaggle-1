import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"allocated={used:.2f}GB, reserved={reserved:.2f}GB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5-0.8B to 4-bit NF4 for edge deployment.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Absolute path to local Qwen3.5-0.8B model directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/runner/work/kaggle-1/kaggle-1/qwen3_5_0_8b_4bit",
        help="Absolute path to output directory for quantized model.",
    )
    parser.add_argument(
        "--prompt",
        default="请用三句话解释什么是模型量化。",
        help="Prompt for generation sanity check.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens for generation. Recommended 64~256 on 4GB VRAM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print("Loading 4-bit quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile skipped: {exc}")

    print("GPU memory:", gpu_mem())

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        end = time.time()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Generation ===")
    print(text)
    print(f"\nLatency: {(end - start):.2f}s")
    print("GPU memory after generation:", gpu_mem())

    print(f"Saving to: {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
