from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import random_split
import math
import glob
import os
from datetime import datetime

# Load and combine data
data1 = pd.read_csv("Interview_Data_6K.csv")
data2 = pd.read_csv("Synthetic_Data_10K.csv")
combined_df = pd.concat([data1, data2], axis=0, ignore_index=True)

# Remove null values
combined_df = combined_df.dropna(subset=['input'])

# Store the common instruction
INSTRUCTION = combined_df['instruction'].iloc[0]

def get_prompt_length(instruction, user_input, tokenizer):
    """Calculate the length of the prompt part (reusable)"""
    prompt = f"{instruction}\n\nUser: {user_input}\nAssistant:"
    encoding = tokenizer(
        prompt,
        truncation=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    return encoding['input_ids'].shape[1]

class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.instruction = INSTRUCTION
        self.input = data['input'].values
        self.output = data['output'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.input)
        
    def __getitem__(self, idx):
        # Construct input text
        input_text = f"{self.instruction}\n\nUser: {self.input[idx]}\nAssistant:"
        
        # Construct full text
        full_text = f"{input_text} {self.output[idx]}"
        
        # Tokenize full text
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize input part（No padding）
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        # create labels
        labels = full_encoding['input_ids'].squeeze().clone()

        '''#Debugging lines
        if idx == 0:  # 只打印第一个样本
            print(f"\n=== SHAPE INFO ===")
            print(f"input_encoding shape: {input_encoding['input_ids'].shape}")
            print(f"Is 2D? {input_encoding['input_ids'].dim() == 2}")
            print(f"Batch size: {input_encoding['input_ids'].shape[0]}")
            print(f"Sequence length: {input_encoding['input_ids'].shape[1]}")
            print("==================\n")'''

        # Set input labels to -10-
        input_length = get_prompt_length(self.instruction, self.input[idx], self.tokenizer)
        labels[:input_length] = -100
        
        # Debugging lines
        '''
        if idx == 0: 
            print(f"Sample {idx}:")
            print(f"Full text length: {len(full_encoding['input_ids'].squeeze())}")
            print(f"Input length: {input_length}")
            print(f"Output tokens to learn: {len(labels) - input_length}")
            print(f"First few tokens of output: {self.tokenizer.decode(labels[input_length:input_length+10])}")
        '''
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("lzw1008/Emollama-7b")
tokenizer.pad_token = tokenizer.eos_token

# Create dataset
dataset = ConversationDataset(combined_df, tokenizer, max_length=512)

# Split into train/eval
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(
        "lzw1008/Emollama-7b",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.enable_input_require_grads()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Prepare model for training
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./emollama-mental-health-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=400,
    eval_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-5,
    bf16=True,  # Change fp16=True to bf16=True
    optim="paged_adamw_8bit",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    gradient_checkpointing=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

checkpoint_dir = "./emollama-mental-health-finetuned"
checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

if checkpoints:
    # Get the checkpoint with the highest number
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    # Resume training from the latest checkpoint
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("Starting training from scratch...")
    # Start training
    trainer.train()

# Save the final model with a timestamp to avoid overwrites
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_model_path = f"emollama-mental-health-lora_{timestamp}"

print(f"\nSaving final model to {final_model_path}")
trainer.model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Optionally, create a symbolic link to the latest version
latest_link = "emollama-mental-health-lora_latest"
if os.path.exists(latest_link):
    os.remove(latest_link)
os.symlink(final_model_path, latest_link)

print("\nRunning evaluation...")

# Evaluate on the entire eval dataset
eval_results = trainer.evaluate()

# Print evaluation metrics
print("\nEvaluation Results:")
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
print(f"Eval Loss: {eval_results['eval_loss']:.4f}")

# Generate some sample predictions
print("\nGenerating sample predictions...")

def generate_response(input_text):
    # Format the input text
    prompt = f"{INSTRUCTION}\n\nUser: {input_text}\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    input_length = get_prompt_length(INSTRUCTION, input_text, tokenizer)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

# Test on a few examples from eval dataset
print("\nSample Predictions:")
for i in range(min(3, len(eval_dataset))):
    actual_idx = eval_dataset.indices[i]  # 这是在combined_df中的真实索引
    input_text = combined_df.iloc[actual_idx]['input']
    actual_output = combined_df.iloc[actual_idx]['output']
    
    generated = generate_response(input_text)
    
    print(f"\nExample {i+1}:")
    print(f"Input: {input_text[:100]}...")
    print(f"Generated: {generated[:100]}...")
    print(f"Actual: {actual_output[:100]}...")
    print("-" * 80)

# Save evaluation results
with open("evaluation_results.txt", "w") as f:
    f.write(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}\n")
    f.write(f"Eval Loss: {eval_results['eval_loss']:.4f}\n")

print("\nEvaluation complete. Results saved to evaluation_results.txt")