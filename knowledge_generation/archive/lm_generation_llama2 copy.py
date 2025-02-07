import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader


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
def initialize_model(model_id="meta-llama/Llama-3.2-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # tokenizer.pad_token_id = model.config.eos_token_id 
    # model.config.pad_token_id = model.config.eos_token_id
    model = Accelerator.prepare(model)
    return model, tokenizer

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

def generate_knowledge_batch_pipeline(prompts, chat_pipeline):
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
def generate_knowledge_batch(prompts, model, tokenizer):
    system_message = "Provide only specific, concise insights gained from the information provided."
    formatted_prompts = [f"<s>[SYSTEM]\n{system_message}\n</s><s>[USER]\n{prompt}\n</s><s>[ASSISTANT]\n" for prompt in prompts]
    dataset = Dataset.from_dict({"prompts": formatted_prompts})

    # Tokenize dataset (batched for efficiency)
    def tokenize_function(batch):
        return tokenizer(batch["prompts"], padding=True, truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare DataLoader
    dataloader = DataLoader(
        tokenized_dataset.with_format("torch"), 
        batch_size=8, 
        collate_fn=lambda x: {
            key: torch.stack([d[key] for d in x]) for key in x[0]
        }
    )
    # Prepare DataLoader for multi-GPU
    dataloader = Accelerator.prepare(dataloader)
    generated_texts = []
    encoded_representations = []
    # Process batches
    for batch in dataloader:
        print(f"Batch: {batch}/{len(dataloader)}")
        # Move inputs to the correct device
        inputs = {key: batch[key].to(Accelerator.device) for key in batch.keys()}
        
        # Forward pass
        outputs = model(**inputs, output_hidden_states=True)
        
        # Get last layer's hidden states
        last_layer_hidden_states = outputs.hidden_states[-1]
        
        # Mean pooling to get encoded representations
        encoded_representation = last_layer_hidden_states.mean(dim=1)
        encoded_representations.append(encoded_representation)
        
        print(f"Batch Encoded Representations Shape: {encoded_representations.shape}")
        # cleaned_text = outputs.replace("Based on the user's movie viewing history, here are some insights into their preferences:\n\n1. ", "")#response.replace(system_message, "").strip()
        cleaned_text = [outputs.replace("Based on the user's movie viewing history, here are some insights into their preferences:\n\n1. ", "") for outputs in outputs]
        generated_texts.append(cleaned_text)
    
    return generated_texts, encoded_representations
    
def main(user_prompt_file, item_prompt_file, user_knowledge_file, item_knowledge_file, user_encoded_representations_file, item_encoded_representations_file):
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
    # pipeline = initialize_chat_pipeline(model_id="meta-llama/Llama-3.2-3B-Instruct")
    # user_knowledge_list = generate_knowledge_batch(user_prompts_list, pipeline)
    model, tokenizer = initialize_model(model_id="meta-llama/Llama-3.2-3B-Instruct") #models--meta-llama--Llama-2-7b-chat-hf
    user_knowledge_list, user_encoded_representations = generate_knowledge_batch(user_prompts_list, model, tokenizer)
    for user_id, user_knowledge in zip(user_ids, user_knowledge_list):
        user_knowledge_dict[user_id] = {
            "prompt": user_prompts[user_id],
            "answer": user_knowledge
        }

    # Generate knowledge for item prompts in batches
    item_ids = list(item_prompts.keys())#[:2]
    item_prompts_list = list(item_prompts.values()) #[:2]
    # item_knowledge_list = generate_knowledge_batch(item_prompts_list, pipeline)
    item_knowledge_list, item_encoded_representations = generate_knowledge_batch(item_prompts_list, model, tokenizer)
    for item_id, item_knowledge in zip(item_ids, item_knowledge_list):
        item_knowledge_dict[item_id] = {
            "prompt": item_prompts[item_id],
            "answer": item_knowledge
        }

    # Save to files
    save_to_file(user_knowledge_file, user_knowledge_dict)
    save_to_file(item_knowledge_file, item_knowledge_dict)
    save_to_file(user_encoded_representations_file, user_encoded_representations)
    save_to_file(item_encoded_representations_file, item_encoded_representations)

    print("Knowledge files generated successfully.")

if __name__ == "__main__":
    DATA_DIR = ''
    PROCESSED_DIR = os.path.join(DATA_DIR, 'ml-1m', 'proc_data')
    USER_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.hist')
    ITEM_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.item')
    KNOWLEDGE_DIR = os.path.join(PROCESSED_DIR, 'knowledge_llama3_2')
    USER_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_DIR, 'user.klg')
    ITEM_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_DIR, 'item.klg')
    USER_ENCODED_REPRESENTATIONS_FILE = os.path.join(KNOWLEDGE_DIR, 'user.enc')
    ITEM_ENCODED_REPRESENTATIONS_FILE = os.path.join(KNOWLEDGE_DIR, 'item.enc')
    main(USER_PROMPT_FILE, ITEM_PROMPT_FILE, USER_KNOWLEDGE_FILE, ITEM_KNOWLEDGE_FILE, USER_ENCODED_REPRESENTATIONS_FILE, ITEM_ENCODED_REPRESENTATIONS_FILE)
