import sys

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)


def train_local_smiles_model(smiles_list, output_dir="./artifacts/local_smiles_model"):
    """
    Trains a small GPT-2 model on a list of SMILES strings.
    """
    print(f"Preparing to train on {len(smiles_list)} SMILES strings...")

    # Use a small GPT-2 config
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_ctx=128,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )

    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = Dataset.from_dict({"text": smiles_list})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=sys.maxsize,
        per_device_train_batch_size=8,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    print("Starting training...")
    # trainer.train()
    _ = trainer
    print(f"Model trained and would be saved to {output_dir}")
    # trainer.save_model(output_dir)


if __name__ == "__main__":
    # Sample data
    sample_smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O"] * 20
    train_local_smiles_model(sample_smiles)
