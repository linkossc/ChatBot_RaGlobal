# app/services/clean_messages.py

import pandas as pd
import json
import os
from config import MESSAGES_CSV


def parse_payload(payload_str):
    """Parse the JSON payload safely"""
    if not isinstance(payload_str, str) or not payload_str.strip():
        return "[Vide]"

    try:
        # Fix double quotes
        payload_str = payload_str.replace('""', '"')
        data = json.loads(payload_str)

        # Case 1: Text message
        if data.get("type") == "text":
            return data.get("text", "").strip()

        # Case 2: Attachment
        elif data.get("type") == "attachment":
            file_name = data.get("attachment", {}).get("fileName", "Fichier")
            return f"[Pièce jointe] {file_name}"

        # Case 3: Reaction or unsupported
        elif "reaction" in str(data) or data.get("type") == "unsupported":
            return "[Réaction]"

        # Default
        else:
            return "[Message non texte]"

    except Exception as e:
        return "[Erreur parsing]"

def clean_messages():
    print("🧹 Nettoyage de messages-csv.csv...")

    if not os.path.exists(MESSAGES_CSV):
        print(f"❌ Fichier introuvable : {MESSAGES_CSV}")
        return

    # Read CSV
    df = pd.read_csv(
        MESSAGES_CSV,
        header=None,
        dtype=str,
        on_bad_lines='skip'
    )

    # Define columns
    columns = [
        'timestamp', 'conversation_id', 'sender_type', 'sender_id',
        'message_id', 'message_type', 'direction', 'payload', 'recipient_id'
    ]
    df = df.iloc[:, :9]  # Keep only 9 columns
    df.columns = columns

    # Clean
    df.dropna(subset=['message_id'], inplace=True)
    df['text'] = df['payload'].apply(parse_payload)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Convert to string for JSON
    df['timestamp'] = df['timestamp'].astype(str).replace('NaT', '')

    # Save
    output_path = os.path.join(os.path.dirname(MESSAGES_CSV), '..', 'processed', 'messages_clean.json')
    df.to_json(output_path, orient='records', indent=2, force_ascii=False)
    print(f"✅ messages_clean.json généré dans {output_path}")
    print(f"📊 {len(df)} messages nettoyés")


if __name__ == "__main__":
    clean_messages()