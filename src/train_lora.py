import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # For brevity, config parsing omitted – use YAML (Hydra or OmegaConf)
    model_name = "meta-llama/Llama-3-8B-Instruct"
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")  # replace with domain dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])
    model = get_peft_model(model, lora_config)

    def tokenize(example):
        return tokenizer(example["instruction"] + example["input"], truncation=True)
    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        max_steps=500,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

if __name__ == "__main__":
    main()