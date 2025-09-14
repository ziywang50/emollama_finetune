from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check GPU availability
if torch.cuda.is_available():
    print("CUDA is available! GPU name:", torch.cuda.get_device_name(0))
    print("Total memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
else:
    print("CUDA not available. Running on CPU.")

# Use 4-bit Emollama 7B for testing
model_name = "lzw1008/Emollama-7b"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

prompt = "Hello, I am testing EmoLLaMA. Can you say something?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
