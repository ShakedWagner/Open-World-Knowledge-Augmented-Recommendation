import json
import os
import transformers
import torch
from datasets import Dataset

def initialize_text_generation_pipeline(model_id="meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto"):
    """
    Initializes the text generation pipeline with the specified model and settings.

    Args:
        model_id (str): The identifier of the model to use.
        torch_dtype: The data type for PyTorch tensors.
        device_map (str): The device map setting for the pipeline.

    Returns:
        pipeline: A configured text generation pipeline.
    """
    # Initialize tokenizer first
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side='left')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Initialize the text generation pipeline with the configured tokenizer and model
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map
    )

    return pipeline



def generate_knowledge(prompt, pipeline):
    # Generate text using the pipeline
    generated_text = pipeline(prompt, max_new_tokens=256, return_full_text=False)[0]['generated_text']
    return generated_text.strip()

def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(json.dumps(content, indent=2))

def load_prompts(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def generate_knowledge_batch(prompts, pipeline):
    """
    Generate knowledge for a batch of prompts.
    """
    # Add system prompt to each user prompt
    system_prompt = "Provide only specific, concise insights gained from the information provided.\n\n"
    formatted_prompts = [system_prompt + prompt for prompt in prompts]
    
    # Create a dataset from the prompts
    dataset = Dataset.from_dict({"prompt": formatted_prompts})
    
    # Generate text using the pipeline in batches
    results = pipeline(
        dataset["prompt"], 
        max_new_tokens=256, 
        return_full_text=False,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,  # You might want to lower this to get more focused responses
        top_p=0.9,
        num_return_sequences=1
    )

    # Extract generated text with error handling
    generated_texts = []
    for result in results:
        print("Debug - Raw result:", result)
        
        if isinstance(result, list):
            generated_text = result[0].get('generated_text', '').strip()
        else:
            generated_text = result.get('generated_text', '').strip()
        
        generated_texts.append(generated_text)

    return generated_texts

def main(user_prompt_file, item_prompt_file, user_knowledge_file, item_knowledge_file):
    # Load prompts from JSON files
    user_prompts = load_prompts(user_prompt_file)
    item_prompts = load_prompts(item_prompt_file)

    # Create dictionaries for user and item knowledge
    user_knowledge_dict = {}
    item_knowledge_dict = {}

    # Generate knowledge for user prompts in batches
    user_ids = list(user_prompts.keys())[:2]
    user_prompts_list = list(user_prompts.values())[:2]
    # Initialize the pipeline
    pipeline = initialize_text_generation_pipeline()
    user_knowledge_list = generate_knowledge_batch(user_prompts_list, pipeline)
    for user_id, user_knowledge in zip(user_ids, user_knowledge_list):
        user_knowledge_dict[user_id] = {
            "prompt": user_prompts[user_id],
            "answer": user_knowledge
        }

    # Generate knowledge for item prompts in batches
    item_ids = list(item_prompts.keys())
    item_prompts_list = list(item_prompts.values())
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
    DATA_DIR = ''
    PROCESSED_DIR = os.path.join(DATA_DIR, 'ml-1m', 'proc_data')
    USER_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.hist')
    ITEM_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.item')
    USER_KNOWLEDGE_FILE = os.path.join(PROCESSED_DIR, 'user.klg')
    ITEM_KNOWLEDGE_FILE = os.path.join(PROCESSED_DIR, 'item.klg')

    main(USER_PROMPT_FILE, ITEM_PROMPT_FILE, USER_KNOWLEDGE_FILE, ITEM_KNOWLEDGE_FILE)