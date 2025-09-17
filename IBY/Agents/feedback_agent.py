# agents/feedback_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re

# Path to your finetuned Llama model adapters
FINETUNED_MODEL_PATH = "Finetuning/finetuned_model" 
# The original Llama model you fine-tuned
BASE_MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"

# Device selection (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None

def load_feedback_model():
    """
    Correctly loads the base Llama model and attaches the finetuned adapters.
    This function should only be called once.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        try:
            # Step 1: Load the base Llama model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"
            )

            # Step 2: Load the tokenizer (from the base model)
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            tokenizer.pad_token = tokenizer.eos_token

            # Step 3: Load the finetuned adapters on top of the base model
            model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
            model.eval()  # Set the model to evaluation mode
            
            print("Finetuned Llama feedback model loaded successfully. ✅")
            return True
        except Exception as e:
            print(f"Error loading model: {e} ❌")
            print("Please ensure your fine-tuned model adapters and base model are in the correct location.")
            model = None
            tokenizer = None
            return False
    return True

def generate_feedback(question, user_answer):
    """
    Generates constructive feedback for a user's answer to an interview question.
    """
    if model is None or tokenizer is None:
        return "Feedback service is currently unavailable. Please load the model first."

    # Craft the prompt to guide the LLM's output
    prompt_template = f"""
    You are an AI mock interview coach. Your task is to provide constructive and detailed feedback on a candidate's answer to an interview question.

    Instructions for feedback:
    1. Start with a positive point about the answer.
    2. Provide a specific area for improvement.
    3. Suggest how the answer could be more complete or structured.
    4. The tone should be encouraging and professional.

    Interview Question: {question}
    Candidate's Answer: {user_answer}

    Feedback:
    """
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract feedback after "Feedback:" if present
    feedback = result.split("Feedback:")[-1].strip()
    return feedback
