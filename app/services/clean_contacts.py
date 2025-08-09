# app/services/clean_contacts.py

import pandas as pd
import json
import os
from config import CONTACTS_CSV

def clean_contacts():
    print("🧹 Nettoyage de contacts.csv...")

    if not os.path.exists(CONTACTS_CSV):
        print(f"❌ Fichier introuvable : {CONTACTS_CSV}")
        return

    # Colonnes attendues
    columns = [
        'ContactID', 'FirstName', 'LastName', 'PhoneNumber', 'Email',
        'Country', 'Language', 'Tags', 'Status', 'Lifecycle',
        'Assignee', 'LastInteractionTime', 'DateTimeCreated',
        'Channels', 'Lead Source', 'State', 'Moyenne Bac',
        'Last Degree', 'Graduation Year', 'Current Degree',
        'Degree Sought', 'Degree Choice', 'Scholarship',
        'University', 'Qualifying URL', 'Eligible', 'Qualifying Score'
    ]

    # Lire le CSV
    try:
        df = pd.read_csv(CONTACTS_CSV, header=None, dtype=str)
        df = df.iloc[:, :len(columns)]  # Garder les bonnes colonnes
        df.columns = columns

        # Nettoyer
        df.fillna('', inplace=True)

        # Convertir les dates avec format explicite
        date_cols = ['LastInteractionTime', 'DateTimeCreated']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                df[col] = df[col].astype(str).replace('NaT', '')

        # ✅ Correction ici : c'est `force_ascii`, pas `ensure_ascii`
        output_path = os.path.join(os.path.dirname(CONTACTS_CSV), '..', 'processed', 'contacts_clean.json')
        df.to_json(output_path, orient='records', indent=2, force_ascii=False)

        print(f"✅ contacts_clean.json généré dans {output_path}")

    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")

if __name__ == "__main__":
    clean_contacts()