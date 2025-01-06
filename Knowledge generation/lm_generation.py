import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the tokenizer and model
model_name = "meta-llama/Llama-3.1-70B"  # Example: Llama-2 7B
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_knowledge(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate text using the model
    outputs = model.generate(**inputs, max_new_tokens=50)
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.strip()

def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(json.dumps(content, indent=2))

def load_prompts(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def main(user_prompt_file, item_prompt_file, user_knowledge_file, item_knowledge_file):
    # Load prompts from JSON files
    user_prompts = load_prompts(user_prompt_file)
    item_prompts = load_prompts(item_prompt_file)

    # Create dictionaries for user and item knowledge
    user_knowledge_dict = {}
    item_knowledge_dict = {}

    # Generate knowledge for each user prompt
    for user_id, prompt in user_prompts.items():
        user_knowledge = generate_knowledge(prompt)
        user_knowledge_dict[user_id] = {
            "prompt": prompt,
            "answer": user_knowledge
        }

    # Generate knowledge for each item prompt
    for item_id, prompt in item_prompts.items():
        item_knowledge = generate_knowledge(prompt)
        item_knowledge_dict[item_id] = {
            "prompt": prompt,
            "answer": item_knowledge
        }

    # Save to files
    save_to_file(user_knowledge_file, user_knowledge_dict)
    save_to_file(item_knowledge_file, item_knowledge_dict)

    print("Knowledge files generated successfully.")

if __name__ == "__main__":
    DATA_DIR = '../data/'
    PROCESSED_DIR = os.path.join(DATA_DIR, 'ml-1m', 'proc_data')
    USER_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.hist')
    ITEM_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.item')
    USER_KNOWLEDGE_FILE = os.path.join(PROCESSED_DIR, 'user.klg')
    ITEM_KNOWLEDGE_FILE = os.path.join(PROCESSED_DIR, 'item.klg')

    main(USER_PROMPT_FILE, ITEM_PROMPT_FILE, USER_KNOWLEDGE_FILE, ITEM_KNOWLEDGE_FILE)