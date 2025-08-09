# app/services/clean_conversations.py

import pandas as pd
import json
import os
from config import CONVERSATIONS_CSV

def clean_conversations():
    print("🧹 Nettoyage de conversations-csv.csv...")

    if not os.path.exists(CONVERSATIONS_CSV):
        print(f"❌ Fichier introuvable : {CONVERSATIONS_CSV}")
        return

    # Colonnes attendues
    columns = [
        'conversation_id', 'start_time', 'end_time', 'contact_id',
        'assignee_id', 'incoming_messages', 'outgoing_messages',
        'last_reply_time', 'status', 'summary', 'last_assignee_id',
        'first_reply_time', 'total_handling_time', 'recipient_id'
    ]

    # Lire le CSV
    try:
        df = pd.read_csv(CONVERSATIONS_CSV, header=None, dtype=str)
        df = df.iloc[:, :len(columns)]  # Garder les bonnes colonnes
        df.columns = columns

        # Nettoyer
        df.fillna('', inplace=True)

        # Convertir les dates avec format explicite
        date_cols = ['start_time', 'end_time', 'last_reply_time', 'first_reply_time']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                df[col] = df[col].astype(str).replace('NaT', '')

        # ✅ Correction ici : c'est `force_ascii`, pas `ensure_ascii`
        output_path = os.path.join(os.path.dirname(CONVERSATIONS_CSV), '..', 'processed', 'conversations_clean.json')
        df.to_json(output_path, orient='records', indent=2, force_ascii=False)

        print(f"✅ conversations_clean.json généré dans {output_path}")

    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")

if __name__ == "__main__":
    clean_conversations()