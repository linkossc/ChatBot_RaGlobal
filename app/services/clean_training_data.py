import json
import os
import time
import requests
from typing import List, Dict, Any, Union

# Les chemins des fichiers sont maintenant définis localement
# from config import INPUT_FILE_PATH_CLEANING, OUTPUT_FILE_PATH_CLEANING

# Définir les chemins des fichiers localement dans le script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE_PATH_CLEANING = os.path.join(BASE_DIR, 'data', 'training', 'training_dataset.json')
OUTPUT_FILE_PATH_CLEANING = os.path.join(BASE_DIR, 'data', 'training', 'cleaned_training_data.json')


# Define a simple exponential backoff strategy for API calls
def backoff_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], max_retries: int = 5,
                    initial_delay: float = 1.0) -> Union[Dict[str, Any], None]:
    """
    Makes a POST request with exponential backoff.
    """
    delay = initial_delay
    for i in range(max_retries):
        try:
            # Added a timeout of 300 seconds to prevent hanging requests
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed (Attempt {i + 1}/{max_retries}): {e}")
            if i < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
    return None


def clean_and_filter_data(conversations: List[Dict[str, Any]], batch_size: int = 5):
    """
    Utilise l'API Gemini pour nettoyer et filtrer les messages d'un ensemble de conversations.
    """
    cleaned_conversations = []
    total_conversations = len(conversations)
    print(f"Début du nettoyage et du filtrage de {total_conversations} conversations en lots de {batch_size}...")

    # --- Set api_key to an empty string so the Canvas environment can inject the correct key ---
    api_key = "AIzaSyCyJsUa68uEt_mw3wsknLljpJAwXR4E93Y"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    for i in range(0, total_conversations, batch_size):
        batch = conversations[i:i + batch_size]
        batch_text = json.dumps(batch, ensure_ascii=False)
        print(f"Traitement du lot {i // batch_size + 1}...")

        prompt = f"""
        Vous êtes un outil de nettoyage de données de conversation. Votre tâche est de lire un ensemble de conversations JSON et de les renvoyer dans un format JSON identique, mais avec les modifications suivantes :

        1.  **Supprimer les messages non clairs ou non pertinents :** Cela inclut les messages contenant uniquement des caractères spéciaux comme '*', les messages vides, les messages d'erreurs, ou tout ce qui n'est pas une réponse lisible.
        2.  **Corriger le format des messages d'erreur :** Si un message a un texte comme "[Erreur parsing]", il doit être supprimé.
        3.  **Conserver les conversations claires :** La conversation doit conserver sa structure (statut, résumé, messages). Si une conversation ne contient que des messages non pertinents, elle doit être supprimée du résultat final.

        Voici le lot de conversations à nettoyer :
        {batch_text}

        Veuillez retourner le JSON nettoyé.
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
                            "conversation_id": {"type": "STRING"},
                            "start_time": {"type": "STRING"},
                            "end_time": {"type": "STRING"},
                            "contact_id": {"type": "STRING"},
                            "assignee_id": {"type": "STRING"},
                            "incoming_messages": {"type": "STRING"},
                            "outgoing_messages": {"type": "STRING"},
                            "last_reply_time": {"type": "STRING"},
                            "status": {"type": "STRING"},
                            "summary": {"type": "STRING"},
                            "last_assignee_id": {"type": "STRING"},
                            "first_reply_time": {"type": "STRING"},
                            "total_handling_time": {"type": "STRING"},
                            "recipient_id": {"type": "STRING"},
                            "messages": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "timestamp": {"type": "STRING"},
                                        "sender_type": {"type": "STRING"},
                                        "text": {"type": "STRING"}
                                    },
                                    "propertyOrdering": ["timestamp", "sender_type", "text"]
                                }
                            }
                        },
                        "propertyOrdering": [
                            "conversation_id", "start_time", "end_time", "contact_id",
                            "assignee_id", "incoming_messages", "outgoing_messages",
                            "last_reply_time", "status", "summary", "last_assignee_id",
                            "first_reply_time", "total_handling_time", "recipient_id",
                            "messages"
                        ]
                    }
                }
            }
        }

        generated_data = backoff_request(api_url, {'Content-Type': 'application/json'}, payload)

        if generated_data and generated_data.get('candidates'):
            try:
                new_conversations = json.loads(generated_data['candidates'][0]['content']['parts'][0]['text'])
                print(f"Lot {i // batch_size + 1} traité avec succès.")
                # Ajoutez les conversations qui ont encore des messages
                for conv in new_conversations:
                    if conv.get('messages'):
                        cleaned_conversations.append(conv)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Erreur lors de l'analyse de la réponse de l'API pour un lot : {e}")
                break
        else:
            print(f"Échec de l'obtention des données de l'API pour un lot. Aucune réponse valide reçue.")
            break

        # Attendre un peu avant de traiter le prochain lot
        time.sleep(5)

    return cleaned_conversations


def run():
    """Point d'entrée pour le nettoyage des données de conversation."""
    try:
        with open(INPUT_FILE_PATH_CLEANING, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        cleaned_data = clean_and_filter_data(training_data)

        if cleaned_data:
            with open(OUTPUT_FILE_PATH_CLEANING, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
            print(f"Données nettoyées sauvegardées dans : {OUTPUT_FILE_PATH_CLEANING}")
            print(f"Nombre de conversations originales : {len(training_data)}")
            print(f"Nombre de conversations après nettoyage : {len(cleaned_data)}")
        else:
            print("Aucune donnée nettoyée n'a pu être générée.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier d'entrée '{INPUT_FILE_PATH_CLEANING}' n'a pas été trouvé.")
    except json.JSONDecodeError:
        print(f"Erreur : Échec du décodage JSON du fichier d'entrée '{INPUT_FILE_PATH_CLEANING}'.")


if __name__ == '__main__':
    run()