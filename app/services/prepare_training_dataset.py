import json
import os

# Définir le chemin de base du projet (FlaskProject)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les chemins d'entrée et de sortie
input_file_path = os.path.join(base_dir, 'app', 'data', 'processed', 'merged_data.json')
output_file_path = os.path.join(base_dir, 'app', 'data', 'training', 'training_dataset.json')


def prepare_training_dataset():
    """
    Charge le fichier merged_data.json, filtre les messages pour ne garder que
    les messages texte, puis sauvegarde le résultat dans training_dataset.json.
    """
    print("Début de la préparation du dataset de formation...")

    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 1. Charger le fichier merged_data.json
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Erreur : Le fichier d'entrée '{input_file_path}' n'a pas été trouvé. Erreur : {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON dans '{input_file_path}'. Le fichier est-il corrompu ? Erreur : {e}")
        return

    # 2. Filtrer les messages pour chaque conversation
    training_dataset = []
    for conversation in merged_data:
        # Créer une nouvelle liste de messages pour ne contenir que les messages texte
        clean_messages = []
        # Le champ 'messages' est déjà trié par heure, donc nous n'avons pas besoin
        # de le trier à nouveau. Nous vérifions simplement le type de message.
        for message in conversation.get('messages', []):
            if message.get('message_type') == 'text':
                # On ne garde que les informations pertinentes pour le training
                clean_messages.append({
                    'timestamp': message.get('timestamp'),
                    'sender_type': message.get('sender_type'),
                    'text': message.get('text')
                })

        # S'il y a des messages texte, ajouter la conversation au dataset
        if clean_messages:
            conversation['messages'] = clean_messages
            training_dataset.append(conversation)

    # 3. Sauvegarder le résultat dans training_dataset.json
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(training_dataset, f, indent=4, ensure_ascii=False)
        print(f"Préparation réussie ! Le fichier a été sauvegardé ici : {output_file_path}")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'écriture du fichier. Erreur : {e}")


def run():
    """Point d'entrée pour la préparation automatique du dataset."""
    prepare_training_dataset()


# Exécuter la fonction si le script est lancé directement
if __name__ == '__main__':
    run()