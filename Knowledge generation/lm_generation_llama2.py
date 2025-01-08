import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset

def initialize_chat_pipeline(model_id="meta-llama/Llama-2-7b-chat-hf"):
    """
    Initializes the chat pipeline with the specified model and settings.

    Args:
        model_id (str): The identifier of the model to use.

    Returns:
        pipeline: A configured chat pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer.pad_token_id = model.config.eos_token_id 
    model.config.pad_token_id = model.config.eos_token_id
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return chat_pipeline

def generate_knowledge(prompts, chat_pipeline):
    """
    Generate knowledge for a batch of prompts using a chat-like model.
    """
    system_message = "Provide only specific, concise insights gained from the information provided."

    generated_texts = []
    for user_message in prompts:
        prompt = f"<s>[SYSTEM]\n{system_message}\n</s><s>[USER]\n{user_message}\n</s><s>[ASSISTANT]\n"
        
        # Generate text using the chat pipeline
        response = chat_pipeline(prompt, max_new_tokens=128, do_sample=True)[0]["generated_text"]
        
        # Post-process to strip out any extraneous tokens if needed
        generated_text = response.replace(prompt, "").strip()
        generated_texts.append(generated_text)

    return generated_texts

def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(json.dumps(content, indent=2))

def load_prompts(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def generate_knowledge_batch(prompts, chat_pipeline):
    """
    Generate knowledge for a batch of prompts using a chat-like model.
    """
    system_message = "Provide only specific, concise insights gained from the information provided."
    formatted_prompts = [f"<s>[SYSTEM]\n{system_message}\n</s><s>[USER]\n{prompt}\n</s><s>[ASSISTANT]\n" for prompt in prompts]
    
    # Generate text using the chat pipeline in batches
    results = chat_pipeline(formatted_prompts, max_new_tokens=256, return_full_text=False, batch_size=8)

    # Extract and post-process generated text
    generated_texts = []
    for result in results:
        if len(result) > 0:
            response = result[0]["generated_text"]
        else:
            response = result["generated_text"]
        
        # Simple post-processing heuristic
        cleaned_text = response.replace("Based on the user's movie viewing history, here are some insights into their preferences:\n\n1. ", "")#response.replace(system_message, "").strip()
        
        generated_texts.append(cleaned_text)

    return generated_texts
def main(user_prompt_file, item_prompt_file, user_knowledge_file, item_knowledge_file):
    # Load prompts from JSON files
    user_prompts = load_prompts(user_prompt_file)
    item_prompts = load_prompts(item_prompt_file)

    # Create dictionaries for user and item knowledge
    user_knowledge_dict = {}
    item_knowledge_dict = {}

    # Generate knowledge for user prompts in batches
    user_ids = list(user_prompts.keys())#[:2]
    user_prompts_list = list(user_prompts.values()) #[:2]
    # Initialize the pipeline
    pipeline = initialize_chat_pipeline()
    user_knowledge_list = generate_knowledge_batch(user_prompts_list, pipeline)
    for user_id, user_knowledge in zip(user_ids, user_knowledge_list):
        user_knowledge_dict[user_id] = {
            "prompt": user_prompts[user_id],
            "answer": user_knowledge
        }

    # Generate knowledge for item prompts in batches
    item_ids = list(item_prompts.keys())#[:2]
    item_prompts_list = list(item_prompts.values()) #[:2]
    item_knowledge_list = generate_knowledge_batch(item_prompts_list, pipeline)
    for item_id, item_knowledge in zip(item_ids, item_knowledge_list):
        item_knowledge_dict[item_id] = {
            "prompt": item_prompts[item_id],
            "answer": item_knowledge
        }

    # Save to files
    save_to_file(user_knowledge_file, user_knowledge_dict)
    save_to_file(item_knowledge_file, item_knowledge_dict)

    print("Knowledge files generated successfully.")

if __name__ == "__main__":
    DATA_DIR = '/nvcr/stor/fast/afeldman/data/tests/data'
    PROCESSED_DIR = os.path.join(DATA_DIR, 'ml-1m', 'proc_data')
    USER_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.hist')
    ITEM_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.item')
    USER_KNOWLEDGE_FILE = os.path.join(PROCESSED_DIR, 'user.klg')
    ITEM_KNOWLEDGE_FILE = os.path.join(PROCESSED_DIR, 'item.klg')

    main(USER_PROMPT_FILE, ITEM_PROMPT_FILE, USER_KNOWLEDGE_FILE, ITEM_KNOWLEDGE_FILE)