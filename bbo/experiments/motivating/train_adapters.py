"""LoRA adapter fine-tuning for the motivating example.

Trains n_adapters LoRA adapters on Yahoo Answers data, reusing the base model.
"""

import json
import gc
import torch
from pathlib import Path
from typing import List

from bbo.experiments.motivating.config import MotivatingConfig


def _make_dataset(texts: List[str], tokenizer, max_length: int = 512):
    """Create a torch Dataset from a list of texts for causal LM training."""
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    encodings["labels"] = encodings["input_ids"].clone()
    return TextDataset(encodings)


def train_single_adapter(
    base_model,
    tokenizer,
    texts: List[str],
    adapter_dir: Path,
    config: MotivatingConfig,
):
    """Train a single LoRA adapter and save it.

    Parameters
    ----------
    base_model : PreTrainedModel
        The base model (already on GPU). Will NOT be modified.
    tokenizer : PreTrainedTokenizer
    texts : list of str
        Training texts for this adapter.
    adapter_dir : Path
        Where to save the LoRA weights.
    config : MotivatingConfig
    """
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import TrainingArguments, Trainer

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
    )

    # Wrap base model with LoRA
    model = get_peft_model(base_model, lora_config)

    # Build dataset
    dataset = _make_dataset(texts, tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        learning_rate=config.learning_rate,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save LoRA weights only
    model.save_pretrained(str(adapter_dir))

    # Cleanup: unmerge LoRA from the base model
    del trainer, model, dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Remove training checkpoints dir to save space
    ckpt_dir = adapter_dir / "checkpoints"
    if ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)


def run_train(config: MotivatingConfig):
    """Train all adapters sequentially, with checkpointing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load training data
    train_path = config.data_dir / "adapter_training_sets.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run 'prepare' step first."
        )
    with open(train_path) as f:
        adapter_specs = json.load(f)

    print(f"Training {len(adapter_specs)} adapters with base model {config.base_model}")

    # Load base model and tokenizer once
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    config.adapters_dir.mkdir(parents=True, exist_ok=True)

    for spec in adapter_specs:
        adapter_id = spec["adapter_id"]
        adapter_dir = config.adapters_dir / f"adapter_{adapter_id:03d}"

        # Checkpointing: skip if already trained
        if (adapter_dir / "adapter_config.json").exists():
            print(f"  Adapter {adapter_id:03d} already exists, skipping.")
            continue

        label = spec["label"]
        frac = spec["sensitive_frac"]
        print(f"  Training adapter {adapter_id:03d} "
              f"(label={label}, sensitive_frac={frac:.2f})...")

        train_single_adapter(
            base_model=base_model,
            tokenizer=tokenizer,
            texts=spec["texts"],
            adapter_dir=adapter_dir,
            config=config,
        )

    print("All adapters trained.")
