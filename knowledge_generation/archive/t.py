import json
import os
import transformers
import torch
from datasets import Dataset
from pathlib import Path
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGenerator:
    def __init__(self, model_id: str = "meta-llama/Llama-2-7b-chat-hf"):
        """Initialize the knowledge generator with a specified model."""
        try:
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto"
            )
        except Exception as e:
            logger.error(f"Failed to initialize model pipeline: {e}")
            raise

    def generate_knowledge(self, prompt: str) -> str:
        """Generate knowledge for a single prompt."""
        try:
            generated_text = self.pipeline(
                prompt,
                max_new_tokens=256,
                return_full_text=False,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating knowledge: {e}")
            return ""

    def generate_knowledge_batch(self, prompts: List[str]) -> List[str]:
        """Generate knowledge for a batch of prompts."""
        try:
            dataset = Dataset.from_dict({"prompt": prompts})
            results = self.pipeline(
                dataset["prompt"],
                max_new_tokens=256,
                return_full_text=False,
                batch_size=8,
                do_sample=True,
                temperature=0.7
            )
            return [result['generated_text'].strip() for result in results]
        except Exception as e:
            logger.error(f"Error generating batch knowledge: {e}")
            return [""] * len(prompts)

def save_to_file(filename: str, content: Dict) -> None:
    """Save content to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(content, file, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving to file {filename}: {e}")
        raise

def load_prompts(filename: str) -> Dict:
    """Load prompts from a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Error loading prompts from {filename}: {e}")
        raise

def main(user_prompt_file: str, item_prompt_file: str, 
         user_knowledge_file: str, item_knowledge_file: str):
    """Main function to generate knowledge for users and items."""
    try:
        # Initialize knowledge generator
        generator = KnowledgeGenerator()
        
        # Load prompts
        user_prompts = load_prompts(user_prompt_file)
        item_prompts = load_prompts(item_prompt_file)

        # Generate knowledge for users
        user_ids = list(user_prompts.keys())
        user_prompts_list = list(user_prompts.values())
        user_knowledge_list = generator.generate_knowledge_batch(user_prompts_list)
        user_knowledge_dict = {
            user_id: {
                "prompt": user_prompts[user_id],
                "answer": knowledge
            }
            for user_id, knowledge in zip(user_ids, user_knowledge_list)
        }

        # Generate knowledge for items
        item_ids = list(item_prompts.keys())
        item_prompts_list = list(item_prompts.values())
        item_knowledge_list = generator.generate_knowledge_batch(item_prompts_list)
        item_knowledge_dict = {
            item_id: {
                "prompt": item_prompts[item_id],
                "answer": knowledge
            }
            for item_id, knowledge in zip(item_ids, item_knowledge_list)
        }

        # Save results
        save_to_file(user_knowledge_file, user_knowledge_dict)
        save_to_file(item_knowledge_file, item_knowledge_dict)

        logger.info("Knowledge files generated successfully.")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    # Use environment variables or configuration file for paths
    DATA_DIR = os.getenv('DATA_DIR', '/nvcr/stor/fast/afeldman/data/tests/data')
    PROCESSED_DIR = Path(DATA_DIR) / 'ml-1m' / 'proc_data'
    
    # Ensure directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    USER_PROMPT_FILE = PROCESSED_DIR / 'prompt.hist'
    ITEM_PROMPT_FILE = PROCESSED_DIR / 'prompt.item'
    USER_KNOWLEDGE_FILE = PROCESSED_DIR / 'user.klg'
    ITEM_KNOWLEDGE_FILE = PROCESSED_DIR / 'item.klg'

    main(
        str(USER_PROMPT_FILE),
        str(ITEM_PROMPT_FILE),
        str(USER_KNOWLEDGE_FILE),
        str(ITEM_KNOWLEDGE_FILE)
    )