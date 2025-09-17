# agents/question_generator_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re

# Path to  finetuned Llama model adapters
FINETUNED_MODEL_PATH = "Finetuning/finetuned_model" 
BASE_MODEL_NAME = "meta-llama/Llama-3-8B-Instruct" 

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None

def load_qg_model():
    """
    Correctly loads the base Llama model and the finetuned adapters.
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
            
            print("Finetuned Llama model loaded successfully. ✅")
            return True
        except Exception as e:
            print(f"Error loading model: {e} ❌")
            model = None
            tokenizer = None
            return False
    return True

def generate_questions(skills, num_questions=5):
    """
    Generates interview questions based on extracted skills.
    """
    # Check if the model is loaded
    if model is None or tokenizer is None:
        print("Model not loaded. Please call load_qg_model() first.")
        return []

    # Format the prompt
    prompt = (
        f"Generate {num_questions} interview questions for a candidate with these skills: {', '.join(skills)}"
    )
    
    # Tokenize the prompt and move to the correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate the questions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the generated text
    result_lines = result.split('\n')
    questions = [
        line.strip()
        for line in result_lines
        if line.strip() and not line.lower().startswith("generate")
    ]
    questions = [q for q in questions if "?" in q or re.match(r'^\d+\.', q)]
    

    return questions[:num_questions]
