import json
from llamaapi import LlamaAPI

# Initialize Llama API
llama = LlamaAPI("<your_api_token>")

def generate_knowledge(input_data, prompt_template):
    # Create a prompt by inserting input data into the template
    prompt = prompt_template.format(input_data=input_data)
    
    # Build the API request
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False
    }
    
    # Execute the Request
    response = llama.run(api_request_json)
    generated_text = response['choices'][0]['message']['content'].strip()
    return generated_text

def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(json.dumps(content, indent=2))

def load_movielens_1m():
    users = {}
    items = {}
    ratings = []

    # Load users
    with open('Open-World-Knowledge-Augmented-Recommendation/data/ml-1m/users.dat', 'r') as f:
        for line in f:
            user_id, gender, age, occupation, zip_code = line.strip().split('::')
            users[user_id] = {
                "gender": gender,
                "age": age,
                "occupation": occupation,
                "zip_code": zip_code
            }

    # Load movies
    with open('Open-World-Knowledge-Augmented-Recommendation/data/ml-1m/movies.dat', 'r', encoding='latin-1') as f:
        for line in f:
            movie_id, title, genres = line.strip().split('::')
            items[movie_id] = {
                "title": title,
                "genres": genres.split('|')
            }

    # Load ratings
    with open('Open-World-Knowledge-Augmented-Recommendation/data/ml-1m/ratings.dat', 'r') as f:
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split('::')
            ratings.append({
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
                "timestamp": timestamp
            })

    return users, items, ratings

def main():
    # Load MovieLens-1M dataset
    users, items, ratings = load_movielens_1m()

    # todo: rewrite as in the paper
    user_prompt_template = "Given a user: {input_data}, with ratings: {ratings}, generate a knowledge about the user."
    item_prompt_template = "Introduce a movie: {input_data}"

    # Create dictionaries for users and items
    user_knowledge_dict = {}
    item_knowledge_dict = {}

    # Organize ratings by user
    user_ratings = {}
    for rating in ratings:
        user_id = rating['user_id']
        if user_id not in user_ratings:
            user_ratings[user_id] = []
        user_ratings[user_id].append(rating)

    # Generate knowledge for each user
    for user_id, user_data in users.items():
        user_ratings_list = user_ratings.get(user_id, [])
        user_knowledge = generate_knowledge(user_data, user_prompt_template.format(input_data=user_data, ratings=user_ratings_list))
        user_knowledge_dict[user_id] = {
            "prompt": user_prompt_template.format(input_data=user_data, ratings=user_ratings_list),
            "answer": user_knowledge
        }

    # Generate knowledge for each item
    for item_id, item_data in items.items():
        item_knowledge = generate_knowledge(item_data, item_prompt_template)
        item_knowledge_dict[item_id] = {
            "prompt": item_prompt_template.format(input_data=item_data),
            "answer": item_knowledge
        }

    # Save to files
    save_to_file('user_knowledge.json', user_knowledge_dict)
    save_to_file('item_knowledge.json', item_knowledge_dict)

    print("Knowledge files generated successfully.")

if __name__ == "__main__":
    main()