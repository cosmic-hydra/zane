import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from drug_discovery.data.collector import DataCollector
from drug_discovery.nvidia_support import DEFAULT_NVIDIA_MODEL


def train_nvidia_pubchem(
    model_id: str = DEFAULT_NVIDIA_MODEL,
    output_dir: str = "./artifacts/nvidia_lora_weights",
    limit: int = 50000,
):
    print(f"Collecting up to {limit} records from PubChem...")
    collector = DataCollector()
    df = collector.collect_from_pubchem(query="aspirin", limit=limit)

    if df.empty:
        print("No data collected from PubChem. Falling back to built-in approved drugs...")
        df = collector.collect_approved_drugs()

    if df.empty:
        print("No data collected at all. Exiting.")
        return

    # We only care about the SMILES strings for causal language modeling
    smiles_list = df["smiles"].dropna().tolist()
    print(f"Retrieved {len(smiles_list)} valid SMILES strings. Preparing dataset...")

    dataset = Dataset.from_dict({"text": smiles_list})

    # Setup tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Load model with quantization (BitsAndBytes)
    print("Loading base model with quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    print("Starting training with PEFT/LoRA...")
    trainer.train()

    print(f"Saving LoRA weights to {output_dir}")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    train_nvidia_pubchem()
