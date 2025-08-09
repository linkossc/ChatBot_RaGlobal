import os
import json
import pandas as pd

DATA_PATH = os.path.join("app", "data", "training", "synthetic_conversations.json")
MODEL_DIR = os.path.join("app", "models", "saved")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Charge les données du fichier JSON et les retourne sous forme de DataFrame."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier de données '{DATA_PATH}' n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        print(f"❌ Erreur : Le fichier '{DATA_PATH}' n'est pas un JSON valide.")
        return None

    rows = []
    for conv in data:
        # Concaténer tous les messages d'une conversation pour en faire un seul document texte
        text = " ".join([m["text"] for m in conv["messages"]])
        rows.append({"text": text, "label": conv["status"]})

    df = pd.DataFrame(rows)
    return df