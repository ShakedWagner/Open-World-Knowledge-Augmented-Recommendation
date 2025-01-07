import json
import os
import transformers
import torch

# Initialize the text generation pipeline
model_id = "meta-llama/Llama-3.1-8B"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

def generate_knowledge(prompt):
    # Generate text using the pipeline
    generated_text = pipeline(prompt, max_new_tokens=256, return_full_text=False)[0]['generated_text']
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