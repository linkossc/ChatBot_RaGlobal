import json
import os
import time
import random
import requests
from typing import List, Dict, Any, Union


# Define a simple exponential backoff strategy for API calls
def backoff_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], max_retries: int = 5,
                    initial_delay: float = 1.0) -> Union[Dict[str, Any], None]:
    """
    Makes a POST request with exponential backoff.
    """
    delay = initial_delay
    for i in range(max_retries):
        try:
            # Added a timeout of 60 seconds to prevent hanging requests
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed (Attempt {i + 1}/{max_retries}): {e}")
            if i < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    return None


def augment_synthetic_data(num_to_generate: int = 200, batch_size: int = 10, sample_size: int = 20):
    """
    Reads existing synthetic conversations and real conversations, then uses
    a large language model to generate and append new synthetic conversations
    in smaller batches, with a new random sample for each batch.
    """
    # 1. Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    real_data_path = os.path.join(base_dir, 'app', 'data', 'training', 'cleaned_training_data.json')
    synthetic_data_path = os.path.join(base_dir, 'app', 'data', 'training', 'synthetic_conversations.json')

    print(f"Starting to augment synthetic data by generating {num_to_generate} new conversations...")

    # 2. Load data: We now load the real data and synthetic data separately.
    real_conversations = []
    try:
        with open(real_data_path, 'r', encoding='utf-8') as f:
            real_conversations.extend(json.load(f))
        print(f"Loaded {len(real_conversations)} real conversations from '{real_data_path}'.")
    except FileNotFoundError:
        print(f"Error: The real training data file '{real_data_path}' was not found. Cannot proceed.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{real_data_path}'. Cannot proceed.")
        return

    existing_synthetic_data = []
    try:
        with open(synthetic_data_path, 'r', encoding='utf-8') as f:
            existing_synthetic_data = json.load(f)
        print(f"Loaded {len(existing_synthetic_data)} existing synthetic conversations from '{synthetic_data_path}'.")
    except FileNotFoundError:
        print(f"Warning: No existing synthetic data found at '{synthetic_data_path}'. Starting from scratch.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{synthetic_data_path}'. Re-initializing as empty list.")

    if not existing_synthetic_data:
        existing_synthetic_data = []

    # Check if there's enough real data to sample from
    if not real_conversations:
        print("Error: No real conversations available to use as a basis for generation.")
        return

    # --- FIX: Set api_key to an empty string so the Canvas environment can inject the correct key ---
    api_key = "AIzaSyCyJsUa68uEt_mw3wsknLljpJAwXR4E93Y"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # --- Génération par lots ---
    generated_count = 0
    while generated_count < num_to_generate:
        # Avertissement: la taille du lot a été réduite pour plus de fiabilité
        to_generate_this_batch = min(batch_size, num_to_generate - generated_count)

        # --- NOUVEAU: Création d'un échantillon aléatoire pour CHAQUE LOT ---
        # L'échantillonnage se fait MAINTENANT uniquement à partir de real_conversations
        sample_conversations = random.sample(real_conversations, min(sample_size, len(real_conversations)))
        example_text = "\n\n".join([
            f"Conversation with status '{conv.get('status', 'N/A')}' and summary '{conv.get('summary', 'N/A')}':\n" +
            "\n".join([f" - {msg.get('sender_type')}: {msg.get('text')}" for msg in conv.get('messages', [])])
            for conv in sample_conversations
        ])

        print(f"Sending request to LLM to generate a batch of {to_generate_this_batch} conversations...")

        prompt = f"""
        You are a data generation tool for a chatbot. Your task is to generate {to_generate_this_batch} new, synthetic, but realistic, conversation data based on a provided style and theme.

        The conversations are between a 'contact' (customer) and a 'user' or 'echo' (a sales representative or a chatbot). The language used is Tunisian Arabic, and the themes revolve around student inquiries about educational programs.

        Here are a few examples of real and synthetic conversations to guide your generation. Use these examples to generate a wide variety of new scenarios, statuses, and summaries:

        {example_text}

        Generate {to_generate_this_batch} new, short, and realistic synthetic conversations. Ensure the flow is logical, and the messages are correctly ordered by sender type. The conversations should be formatted as a JSON array of objects, where each object represents a conversation. Each conversation object must contain a 'status', a 'summary', and an array of 'messages'. Each message object must contain a 'sender_type' ('contact', 'user', or 'echo') and 'text'. The 'summary' should be a brief note about the conversation's outcome, and the 'status' should be 'Qualified', 'Unqualified', or 'To follow up'.

        Make sure to write everything in Tunisian Arabic just like the examples.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "status": {"type": "STRING"},
                            "summary": {"type": "STRING"},
                            "messages": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "sender_type": {"type": "STRING"},
                                        "text": {"type": "STRING"}
                                    },
                                    "propertyOrdering": ["sender_type", "text"]
                                }
                            }
                        },
                        "propertyOrdering": ["status", "summary", "messages"]
                    }
                }
            }
        }

        generated_data = backoff_request(api_url, {'Content-Type': 'application/json'}, payload)

        if generated_data and generated_data.get('candidates'):
            try:
                new_synthetic_conversations = json.loads(generated_data['candidates'][0]['content']['parts'][0]['text'])
                print(f"Successfully generated {len(new_synthetic_conversations)} new conversations in this batch.")
                existing_synthetic_data.extend(new_synthetic_conversations)
                generated_count += len(new_synthetic_conversations)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing the LLM response for a batch: {e}")
                break
        else:
            print("Failed to generate data from the LLM. No valid response received for a batch.")
            break

        # Avertissement: la pause entre les lots a été augmentée pour plus de fiabilité
        if generated_count < num_to_generate:
            print("Waiting a moment before the next batch...")
            time.sleep(5)

    print(f"Total conversations generated: {generated_count}")
    # 4. Save the augmented data to the same file
    with open(synthetic_data_path, 'w', encoding='utf-8') as f:
        json.dump(existing_synthetic_data, f, indent=4, ensure_ascii=False)
    print(f"Augmented data has been saved to: {synthetic_data_path}")
    print(f"Total synthetic conversations now: {len(existing_synthetic_data)}")


def run():
    """Entry point for the synthetic data augmentation."""
    # Le nombre total de conversations à générer
    # Vous pouvez modifier le nombre total et la taille du lot ici si vous le souhaitez
    augment_synthetic_data(num_to_generate=200, batch_size=10, sample_size=20)


if __name__ == '__main__':
    run()