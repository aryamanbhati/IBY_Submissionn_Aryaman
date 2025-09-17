# Finetuning/finetuning_script.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import pandas as pd
import os

def run_fine_tuning(dataset_path="./Finetuning/Train.csv", output_dir="./Finetuning/finetuned_model"):
    """
    Orchestrates the fine-tuning pipeline using a CSV dataset.
    """
    # Step 1: Prepare the Dataset
    try:
        df = pd.read_csv(dataset_path)
        
        dataset = load_dataset("csv", data_files=dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}. Please ensure your dataset exists.")
        return

    def format_example(example):
        """Formats the data into an instruction-following prompt."""
        prompt = f"### Interviewer Prompt:\nGenerate interview questions for a candidate with skills: {example['skills']}.\n\n### Generated Questions:\n"
        full_text = prompt + example['output']
        return {"text": full_text}

    formatted_dataset = dataset.map(format_example)

    # Step 2: Load Model and Tokenizer
    model_name = "path/to/your/local/llama-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Step 3: Tokenize the Dataset
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=True, padding="max_length"),
        batched=True
    )

    # Step 4: Configure and Apply LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"]
    )
    
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # Step 5: Train the Model
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=lambda data: {
            'input_ids': torch.stack([d['input_ids'] for d in data]),
            'attention_mask': torch.stack([d['attention_mask'] for d in data]),
            'labels': torch.stack([d['input_ids'] for d in data]),
        }
    )

    print("Starting fine-tuning process...")
    trainer.train()
    print("Fine-tuning complete. Saving LoRA adapters.")

    # Step 6: Save the Fine-Tuned Model Adapters
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer.save_model(output_dir)
    print(f"LoRA adapters saved to {output_dir}")

if __name__ == "__main__":

    run_fine_tuning()
