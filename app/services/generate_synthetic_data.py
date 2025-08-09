import json
import os
import time
import random
import requests


# Define a simple exponential backoff strategy for API calls
def backoff_request(url, headers, payload, max_retries=5, initial_delay=1.0):
    delay = initial_delay
    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed (Attempt {i + 1}/{max_retries}): {e}")
            if i < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    return None


def generate_synthetic_data():
    """
    Reads the training_dataset.json file, uses a large language model to
    generate synthetic conversations based on the themes and style, and
    saves them to a new JSON file.
    """
    # 1. Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file_path = os.path.join(base_dir, 'app', 'data', 'training', 'training_dataset.json')
    output_file_path = os.path.join(base_dir, 'app', 'data', 'training', 'synthetic_conversations.json')

    print("Starting synthetic data generation...")

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 2. Load and analyze existing data
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The input file '{input_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file_path}'. Is the file corrupted?")
        return

    # Extract conversation examples to use as a prompt for the LLM
    # We'll take a few random examples to give the model context
    sample_conversations = random.sample(training_data, min(5, len(training_data)))

    # Format the examples into a string for the prompt
    example_text = "\n\n".join([
        f"Conversation with status '{conv['status']}' and summary '{conv['summary']}':\n" +
        "\n".join([f" - {msg['sender_type']}: {msg['text']}" for msg in conv['messages']])
        for conv in sample_conversations
    ])

    # 3. Use an LLM to generate synthetic conversations
    # You will need to replace "YOUR_API_KEY" with your actual API key.
    # Note: This is a placeholder for demonstration. The system will handle API key injection.
    api_key = "AIzaSyCyJsUa68uEt_mw3wsknLljpJAwXR4E93Y"
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + api_key

    # The prompt asks the LLM to act as a data generator
    prompt = f"""
    You are a data generation tool for a chatbot. Your task is to generate synthetic, but realistic, conversation data based on a provided style and theme.

    The conversations are between a 'contact' (customer) and a 'user' or 'echo' (a sales representative or a chatbot). The language used is Tunisian Arabic, and the themes revolve around student inquiries about educational programs.

    Here are a few examples of real conversations to guide your generation:

    {example_text}

    Generate 5 new, short, and realistic synthetic conversations. Ensure the flow is logical, and the messages are correctly ordered by sender type. The conversations should be formatted as a JSON array of objects, where each object represents a conversation. Each conversation object must contain a 'status', a 'summary', and an array of 'messages'. Each message object must contain a 'sender_type' ('contact', 'user', or 'echo') and 'text'. The 'summary' should be a brief note about the conversation's outcome, and the 'status' should be 'Qualified', 'Unqualified', or 'To follow up'.

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

    print("Sending request to LLM to generate synthetic conversations...")

    generated_data = backoff_request(api_url, {'Content-Type': 'application/json'}, payload)

    if generated_data and generated_data.get('candidates'):
        try:
            # The API returns the JSON as a string, so we need to parse it
            synthetic_conversations = json.loads(generated_data['candidates'][0]['content']['parts'][0]['text'])
            print("Successfully generated synthetic data.")

            # 4. Save the synthetic data to a new JSON file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(synthetic_conversations, f, indent=4, ensure_ascii=False)
            print(f"Synthetic data has been saved to: {output_file_path}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing the LLM response: {e}")
    else:
        print("Failed to generate data from the LLM. No valid response received.")

def run():
    """
    Entry
    point
    for the synthetic data generation."""
    generate_synthetic_data()

if __name__ == '__main__':
    run()
