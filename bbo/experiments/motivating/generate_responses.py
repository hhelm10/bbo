"""Response generation for the motivating example.

Loads each LoRA adapter, generates responses to all queries at temperature 0.
"""

import json
import gc
import torch
from pathlib import Path
from typing import List

from bbo.experiments.motivating.config import MotivatingConfig


def _generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int) -> List[str]:
    """Generate responses for a batch of prompts at temperature 0."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    responses = []
    for i, output in enumerate(outputs):
        prompt_len = inputs["input_ids"][i].shape[0]
        generated = output[prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)

    return responses


def run_generate(config: MotivatingConfig):
    """Generate responses for all adapters × all queries."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load queries
    query_path = config.data_dir / "queries.json"
    if not query_path.exists():
        raise FileNotFoundError(
            f"Queries not found at {query_path}. Run 'prepare' step first."
        )
    with open(query_path) as f:
        queries = json.load(f)

    n_queries = len(queries)
    print(f"Loaded {n_queries} queries")

    print(f"Loading model {config.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    # Format queries using the model's chat template
    query_texts = []
    for q in queries:
        messages = [{"role": "user", "content": q["text"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        query_texts.append(text)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    config.responses_dir.mkdir(parents=True, exist_ok=True)

    # Interleave class-0 and class-1 adapters so we can monitor classification
    # accuracy as more models complete (0, 50, 1, 51, 2, 52, ...)
    n_per_class = config.n_per_class
    adapter_order = []
    for i in range(n_per_class):
        adapter_order.append(i)              # class-0
        adapter_order.append(i + n_per_class) # class-1

    for adapter_id in adapter_order:
        adapter_dir = config.adapters_dir / f"adapter_{adapter_id:03d}"
        response_path = config.responses_dir / f"adapter_{adapter_id:03d}.json"

        # Checkpointing
        if response_path.exists():
            print(f"  Adapter {adapter_id:03d} responses exist, skipping.")
            continue

        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  WARNING: Adapter {adapter_id:03d} not trained, skipping.")
            continue

        print(f"  Generating responses for adapter {adapter_id:03d}...")

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model.eval()

        # Generate in batches
        all_responses = []
        batch_size = config.gen_batch_size
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_prompts = query_texts[start:end]
            batch_responses = _generate_batch(
                model, tokenizer, batch_prompts, config.max_new_tokens
            )
            all_responses.extend(batch_responses)

        # Save responses
        with open(response_path, "w") as f:
            json.dump(all_responses, f)

        # Cleanup LoRA
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("All responses generated.")
