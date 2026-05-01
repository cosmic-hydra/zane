"""Local NVIDIA LLM fine-tuning on public molecule databases."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dependency_audit import audit_missing_modules, format_missing_modules

DEFAULT_NVIDIA_MODEL = "nvidia/Nemotron-3-8B-Base-4k"
SUPPORTED_MOLECULE_SOURCES = ("pubchem", "chembl", "approved_drugs", "drugbank")
CORE_FINE_TUNE_MODULES = ("torch", "datasets", "transformers", "peft", "accelerate", "numpy", "pandas")


def _required_modules_for_sources(sources: Sequence[str] | None) -> list[str]:
    modules = list(CORE_FINE_TUNE_MODULES)
    normalized_sources = {str(source).strip().lower() for source in (sources or [])}

    if "pubchem" in normalized_sources:
        modules.append("pubchempy")
    if "chembl" in normalized_sources:
        modules.append("chembl_webresource_client")

    return modules


def _ensure_runtime_dependencies(sources: Sequence[str] | None) -> None:
    missing = audit_missing_modules(_required_modules_for_sources(sources))
    if missing:
        raise RuntimeError(
            "Missing dependencies for local NVIDIA fine-tuning: "
            f"{format_missing_modules(missing)}. Install requirements with `pip install -r requirements.txt` "
            "or at least the fine-tuning subset before running this script."
        )


def _can_use_4bit() -> bool:
    from importlib.util import find_spec

    import torch

    return torch.cuda.is_available() and find_spec("bitsandbytes") is not None and find_spec("accelerate") is not None


def _load_model_and_tokenizer(model_id: str, trust_remote_code: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if _can_use_4bit():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if _can_use_4bit():
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    return tokenizer, model


def train_nvidia_pubchem(
    model_id: str = DEFAULT_NVIDIA_MODEL,
    output_dir: str = "./artifacts/nvidia_lora_weights",
    sources: Sequence[str] | None = None,
    pubchem_query: str = "drug",
    chembl_target: str | None = None,
    chembl_activity_type: str | None = None,
    limit_per_source: int = 5000,
    drugbank_file: str | None = None,
    fallback_to_approved_drugs: bool = True,
    max_length: int = 256,
    epochs: int = sys.maxsize,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    save_steps: int = 500,
    logging_steps: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    trust_remote_code: bool = False,
):
    _ensure_runtime_dependencies(sources)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    from drug_discovery.data.collector import DataCollector
    from drug_discovery.training.nvidia_llm_finetune import build_training_frame, collect_public_molecule_data

    print(f"Collecting public molecule records from: {', '.join(sources or SUPPORTED_MOLECULE_SOURCES)}")
    collector = DataCollector()
    df = collect_public_molecule_data(
        collector=collector,
        sources=sources,
        pubchem_query=pubchem_query,
        chembl_target=chembl_target,
        chembl_activity_type=chembl_activity_type,
        limit_per_source=limit_per_source,
        drugbank_file=drugbank_file,
        fallback_to_approved_drugs=fallback_to_approved_drugs,
    )

    if df.empty:
        print("No data collected at all. Exiting.")
        return

    report = collector.generate_data_quality_report(df)
    print(
        "Collected "
        f"{report['total_rows']} rows | valid={report['valid_smiles_rows']} | unique={report['unique_smiles']}"
    )

    training_frame = build_training_frame(df)
    if training_frame.empty:
        print("No training texts could be constructed. Exiting.")
        return

    print(f"Prepared {len(training_frame)} text samples for local fine-tuning.")

    dataset = Dataset.from_pandas(training_frame, preserve_index=False)

    tokenizer, model = _load_model_and_tokenizer(model_id, trust_remote_code=trust_remote_code)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Configure LoRA
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(Path(output_dir)),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        bf16=getattr(torch.cuda, "is_bf16_supported", lambda: False)(),
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune an NVIDIA LLM locally on public molecule databases")
    parser.add_argument("--model-id", default=DEFAULT_NVIDIA_MODEL, help="Local path or NVIDIA model id")
    parser.add_argument("--output-dir", default="./artifacts/nvidia_lora_weights")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SUPPORTED_MOLECULE_SOURCES),
        choices=list(SUPPORTED_MOLECULE_SOURCES),
        help="Public molecule sources to include",
    )
    parser.add_argument("--pubchem-query", default="drug", help="PubChem query term")
    parser.add_argument("--chembl-target", default=None, help="Optional ChEMBL target filter")
    parser.add_argument("--chembl-activity-type", default=None, help="Optional ChEMBL activity type filter")
    parser.add_argument("--limit-per-source", type=int, default=5000)
    parser.add_argument("--drugbank-file", default=None, help="Path to a local DrugBank export")
    parser.add_argument("--no-approved-fallback", action="store_true", help="Disable fallback to approved drugs")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=sys.maxsize)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only audit runtime dependencies for the selected sources and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_deps:
        missing = audit_missing_modules(_required_modules_for_sources(args.sources))
        print("Missing deps:", format_missing_modules(missing))
        raise SystemExit(1 if missing else 0)

    train_nvidia_pubchem(
        model_id=args.model_id,
        output_dir=args.output_dir,
        sources=args.sources,
        pubchem_query=args.pubchem_query,
        chembl_target=args.chembl_target,
        chembl_activity_type=args.chembl_activity_type,
        limit_per_source=args.limit_per_source,
        drugbank_file=args.drugbank_file,
        fallback_to_approved_drugs=not args.no_approved_fallback,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        trust_remote_code=args.trust_remote_code,
    )

if __name__ == "__main__":
    main()
