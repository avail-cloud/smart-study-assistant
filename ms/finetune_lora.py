# finetune_lora.py
"""
LoRA fine-tuning demo script (for grading prompts).
Intended to run in Colab with GPU.
Make sure to upload 'sample_data/grade_data.jsonl' to the working dir.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_NAME = "gpt2"  # small and quick for demo
TOKENIZER_NAME = MODEL_NAME
OUTPUT_DIR = "lora_grade_demo"

def load_data(path="sample_data/grade_data.jsonl"):
    ds = load_dataset("json", data_files=path, split="train")
    # Expect fields: prompt, completion OR for our sample we use 'prompt' only with instruction style
    return ds

def tokenize_function(examples, tokenizer):
    texts = [ (ex.get("prompt","") + "\n" + ex.get("completion","")) for ex in examples ]
    return tokenizer(texts, truncation=True, max_length=512)

def main():
    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    ds = load_data()
    def map_fn(ex):
        txt = ex.get("prompt","")
        # if 'completion' exists, append it; else we expect prompt->completion style in file
        return tokenizer(txt, truncation=True, max_length=512)

    tokenized = ds.map(lambda ex: tokenizer(ex["prompt"] + ex.get("completion",""), truncation=True, max_length=512), batched=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=2,
        learning_rate=2e-4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("LoRA fine-tune complete. Saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
