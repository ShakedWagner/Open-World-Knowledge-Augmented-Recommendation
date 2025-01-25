import re
import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge_encoding.utils import save_json
from knowledge_encoding.lm_encoding import remap_item


def initialize_model(model_id="meta-llama/Llama-2-7b-chat-hf"): #meta-llama/Llama-3.2-3B-Instruct
    # Initialize accelerator first
    accelerator = Accelerator()
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'  # Set padding to the left side
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Ensure the pad_token_id is set
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to 'left' for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    return model, tokenizer, accelerator


def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(json.dumps(content, indent=2))

def load_prompts(filename):
    with open(filename, 'r') as file:
        return json.load(file)
def extract_assistant_message(text, prompt):
    """
    Extracts everything between <s>[ASSISTANT] ... </s> 
    by splitting on known markers.
    Returns None if the markers aren't found.
    """
    # parts = text.split("[ASSISTANT]")
    # if len(parts) > 1:
    #     text = parts[1].strip()
    text = text.replace(prompt, "")
    pattern = r'[^A-Za-z0-9 .,\?!;():]+/'
    cleaned = re.sub(pattern, '', text)
    return cleaned
    
def generate_knowledge_batch(prompts, model, tokenizer, accelerator):
    """
    Generate knowledge for a batch of prompts using accelerator.
    """
    # system_message = "Provide only specific, concise insights gained from the information provided."
    # formatted_prompts = [f"<s>[SYSTEM]\n{system_message}\n</s><s>[USER]\n{prompt}\n</s><s>[ASSISTANT]\n" 
    #                     for prompt in prompts]
    
    # Create dataset
    # dataset = Dataset.from_dict({"prompts": formatted_prompts})
    dataset = Dataset.from_dict({"prompts": prompts})
    # Tokenize function
    def tokenize_function(batch):
        return tokenizer(
            batch["prompts"], 
            padding="max_length",  # Ensure padding is applied
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset.with_format("torch"),
        batch_size=64,
        shuffle=False
    )
    dataloader = accelerator.prepare(dataloader)
    generated_texts = []
    encoded_representations = []
    
    # Process batches
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Prepare inputs
            inputs = {
                'input_ids': batch['input_ids'].to(model.device),
                'attention_mask': batch['attention_mask'].to(model.device)
            }
            
            # Generate outputs
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # return_dict_in_generate=True,
                    # output_hidden_states=True
                )
                model_outputs = model(**inputs, output_hidden_states=True)
            # Decode outputs
            decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            final_messages = []
            for i, decoded in enumerate(decoded_texts):
                assistant_only = extract_assistant_message(decoded, dataset[i]["prompts"])
                final_messages.append(assistant_only)# generated_hidden_states = outputs.hidden_states
            # final_layer_hidden_states = generated_hidden_states[-1]
            # batch_encodings = final_layer_hidden_states.mean(dim=1).cpu()

            # Get encoded representations from the last hidden states
            
            last_hidden_states = model_outputs.hidden_states[-1]
            batch_encodings = last_hidden_states.mean(dim=1)
            encoded_representations.append(batch_encodings.cpu())
            
            # Clean outputs

            generated_texts.extend(final_messages)
    
    # Concatenate all encoded representations
    encoded_representations = torch.cat(encoded_representations, dim=0)
    
    return generated_texts, encoded_representations
 

def main(user_prompt_file, item_prompt_file, user_knowledge_file, item_knowledge_file, 
         user_encoded_representations_file, item_encoded_representations_file):
    # Load prompts
    user_prompts = load_prompts(user_prompt_file)
    item_prompts = load_prompts(item_prompt_file)

    # Initialize model
    model, tokenizer, accelerator = initialize_model("meta-llama/Llama-2-7b-chat-hf") #meta-llama/Llama-3.2-3B-Instruct")
    
    # Process user prompts
    user_ids = list(user_prompts.keys())
    user_prompts_list = list(user_prompts.values())
    user_knowledge_list, user_encoded_representations = generate_knowledge_batch(
        user_prompts_list, model, tokenizer, accelerator
    )
    
    # Create user knowledge dictionary
    user_knowledge_dict = {
        user_id: {
            "prompt": user_prompts[user_id],
            "ans": knowledge
        }
        for user_id, knowledge in zip(user_ids, user_knowledge_list)
    }

    # Process item prompts
    item_ids = list(item_prompts.keys())
    item_prompts_list = list(item_prompts.values())
    item_knowledge_list, item_encoded_representations = generate_knowledge_batch(
        item_prompts_list, model, tokenizer, accelerator
    )
    
    # Create item knowledge dictionary
    item_knowledge_dict = {
        item_id: {
            "prompt": item_prompts[item_id],
            "ans": knowledge
        }
        for item_id, knowledge in zip(item_ids, item_knowledge_list)
    }

    # Convert tensors to lists
    user_encoded_representations_list = user_encoded_representations.cpu().tolist()
    item_encoded_representations_list = item_encoded_representations.cpu().tolist()

    # Map indices to vectors
    user_vec_dict = remap_item(user_ids, user_encoded_representations_list)
    item_vec_dict = remap_item(item_ids, item_encoded_representations_list)

    # Save results as JSON
    os.makedirs(os.path.dirname(user_knowledge_file), exist_ok=True)
    save_to_file(user_knowledge_file, user_knowledge_dict)
    save_to_file(item_knowledge_file, item_knowledge_dict)
    save_json(user_vec_dict, user_encoded_representations_file)
    save_json(item_vec_dict, item_encoded_representations_file)

    print("Knowledge files generated successfully.")

if __name__ == "__main__":
    DATA_DIR = '/nvcr/stor/fast/afeldman/data/tests/data'
    PROCESSED_DIR = os.path.join(DATA_DIR, 'ml-1m', 'proc_data_w_cf') # proc_data
    USER_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.hist')
    ITEM_PROMPT_FILE = os.path.join(PROCESSED_DIR, 'prompt.item')
    
    KNOWLEDGE_DIR = os.path.join(DATA_DIR, 'ml-1m', 'knowledge_w_cf', 'Llama-2-7b-chat-hf')
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
    USER_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_DIR, 'user.klg')
    ITEM_KNOWLEDGE_FILE = os.path.join(KNOWLEDGE_DIR, 'item.klg')
    USER_ENCODED_REPRESENTATIONS_FILE = os.path.join(KNOWLEDGE_DIR, 'user.enc')
    ITEM_ENCODED_REPRESENTATIONS_FILE = os.path.join(KNOWLEDGE_DIR, 'item.enc')
    main(USER_PROMPT_FILE, ITEM_PROMPT_FILE, USER_KNOWLEDGE_FILE, ITEM_KNOWLEDGE_FILE, USER_ENCODED_REPRESENTATIONS_FILE, ITEM_ENCODED_REPRESENTATIONS_FILE)
