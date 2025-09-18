#!/usr/bin/env python3
"""
Comprehensive test script for EmoLLaMA Mental Health Model
Tests various mental health scenarios and displays full responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from datetime import datetime
import os

# Configuration
MODEL_PATH = "emollama-mental-health-lora_latest"
BASE_MODEL = "lzw1008/Emollama-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load instruction from training data (if available)
try:
    data = pd.read_csv("Synthetic_Data_10K.csv", nrows=1)
    INSTRUCTION = data['instruction'].iloc[0]
except:
    INSTRUCTION = "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description."

print(f"Using device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")
print(f"Base model: {BASE_MODEL}")
print("-" * 80)

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
).to(DEVICE)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()
print("Model loaded successfully!")
print("-" * 80)

def generate_response(input_text, max_new_tokens=300, temperature=0.7, top_p=0.9):
    """Generate a response for the given input"""
    # Format prompt exactly as in training
    prompt = f"{INSTRUCTION}\n\nUser: {input_text}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    prompt_length = inputs["input_ids"].shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=30,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response

# Test cases covering various mental health scenarios
test_cases = [
    {
        "category": "Anxiety",
        "inputs": [
            "I've been feeling very anxious about my upcoming job interview. My heart races and I can't sleep.",
            "I'm experiencing panic attacks when I'm in crowded places. What should I do?",
            "My anxiety is affecting my daily life. I can't concentrate on work anymore."
        ]
    },
    {
        "category": "Depression",
        "inputs": [
            "I've been feeling hopeless and empty for weeks. Nothing brings me joy anymore.",
            "I don't have the energy to get out of bed most days. Is this depression?",
            "I've lost interest in all my hobbies and don't want to see friends anymore."
        ]
    },
    {
        "category": "Relationship Issues",
        "inputs": [
            "My partner and I keep having the same arguments. How can we communicate better?",
            "I'm struggling with trust issues after being betrayed in my last relationship.",
            "I feel lonely even when I'm with my family. What's wrong with me?"
        ]
    },
    {
        "category": "Stress Management",
        "inputs": [
            "Work stress is overwhelming me. I'm working 60 hours a week and feel burned out.",
            "I'm a student facing exam pressure and I don't know how to cope.",
            "Taking care of my elderly parents while working full-time is exhausting me."
        ]
    },
    {
        "category": "Grief and Loss",
        "inputs": [
            "I lost my mother six months ago and I still cry every day. Is this normal?",
            "My best friend passed away suddenly and I don't know how to process this grief.",
            "After my divorce, I feel like I'm grieving the life I thought I would have."
        ]
    },
    {
        "category": "Self-Esteem",
        "inputs": [
            "I constantly compare myself to others and always feel inadequate.",
            "I can't accept compliments and always think people are just being nice.",
            "My self-confidence is so low that I avoid social situations."
        ]
    }
]

# Run tests
print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL TESTING")
print("=" * 80)

results = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for category_data in test_cases:
    category = category_data["category"]
    print(f"\n\n{'='*80}")
    print(f"CATEGORY: {category}")
    print(f"{'='*80}")
    
    for i, user_input in enumerate(category_data["inputs"], 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}:")
        print(f"User: {user_input}")
        print(f"\nAssistant: ", end="", flush=True)
        
        # Generate response
        response = generate_response(user_input)
        print(response)
        
        # Store result
        results.append({
            "category": category,
            "input": user_input,
            "response": response,
            "timestamp": datetime.now()
        })
        
        # Check for common issues
        if len(response.strip()) < 50:
            print("\n⚠️  Warning: Response seems too short")
        if "You are a helpful mental health" in response:
            print("\n❌ Error: Still generating system prompt!")
        if response.count(response.split()[0]) > 3:
            print("\n⚠️  Warning: Possible repetition detected")

# Additional edge case tests
print(f"\n\n{'='*80}")
print("EDGE CASE TESTING")
print(f"{'='*80}")

edge_cases = [
    "Help",
    "I want to end it all",  # Crisis situation
    "ajsdkfj asdlfkj asldkfj",  # Gibberish
    "Tell me a joke",  # Off-topic
    "What's 2+2?",  # Non-mental health
    "I'm perfect and have no problems",  # Denial
    "My therapist said I should talk to you",  # Meta-reference
]

for i, edge_input in enumerate(edge_cases, 1):
    print(f"\n{'-'*80}")
    print(f"Edge Test {i}:")
    print(f"User: {edge_input}")
    print(f"\nAssistant: ", end="", flush=True)
    
    response = generate_response(edge_input, max_new_tokens=300)
    print(response)
    
    results.append({
        "category": "Edge Cases",
        "input": edge_input,
        "response": response,
        "timestamp": datetime.now()
    })

# Parameter variation testing
print(f"\n\n{'='*80}")
print("PARAMETER VARIATION TESTING")
print(f"{'='*80}")

test_input = "I'm feeling overwhelmed with life and don't know where to start getting help."

parameters = [
    {"temp": 0.3, "top_p": 0.8, "desc": "More focused/deterministic"},
    {"temp": 0.7, "top_p": 0.9, "desc": "Balanced (default)"},
    {"temp": 0.9, "top_p": 0.95, "desc": "More creative/varied"},
]

for params in parameters:
    print(f"\n{'-'*80}")
    print(f"Parameters: Temperature={params['temp']}, Top-p={params['top_p']} ({params['desc']})")
    print(f"User: {test_input}")
    print(f"\nAssistant: ", end="", flush=True)
    
    response = generate_response(test_input, temperature=params['temp'], top_p=params['top_p'])
    print(response)

# Save results
output_file = f"test_results_{timestamp}.csv"
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False)
print(f"\n\n{'='*80}")
print(f"Test results saved to: {output_file}")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"Total tests run: {len(results)}")
print(f"Average response length: {df_results['response'].str.len().mean():.0f} characters")
print(f"Shortest response: {df_results['response'].str.len().min()} characters")
print(f"Longest response: {df_results['response'].str.len().max()} characters")

# Check for quality issues
issues = []
for idx, row in df_results.iterrows():
    if len(row['response'].strip()) < 50:
        issues.append(f"Short response for: {row['input'][:50]}...")
    if "You are a helpful" in row['response']:
        issues.append(f"System prompt in response for: {row['input'][:50]}...")

if issues:
    print(f"\n⚠️  Found {len(issues)} potential issues:")
    for issue in issues[:5]:  # Show first 5
        print(f"  - {issue}")
else:
    print("\n✅ No major issues detected!")

print(f"\n{'='*80}")
print("Testing complete!")
print(f"{'='*80}")

# Load ONLY the base model (no LoRA)
print("Loading original Emollama-7b for comparison...")
original_model = AutoModelForCausalLM.from_pretrained(
    "lzw1008/Emollama-7b",
    torch_dtype=torch.float16,
).to(DEVICE)

# Test with same prompts
test_prompts = [
    "What's 2+2?",
    "Tell me a joke",
    "I'm feeling anxious about my job interview"
]

for prompt in test_prompts:
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt}")
    
    # Original model response
    inputs = tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt").to(DEVICE)
    original_output = original_model.generate(**inputs, max_new_tokens=300)
    original_response = tokenizer.decode(original_output[0], skip_special_tokens=True)
    
    # Your fine-tuned response  
    finetuned_response = generate_response(prompt)
    
    print(f"\nOriginal: {original_response}")
    print(f"\nFinetuned: {finetuned_response}")