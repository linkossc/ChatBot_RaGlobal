import json
import os

# Définir le chemin de base du projet (FlaskProject)
# En supposant que le script est dans app/services/
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les chemins d'entrée et de sortie en utilisant la structure de votre projet
conversations_file_path = os.path.join(base_dir, 'app', 'data', 'processed', 'conversations_clean.json')
messages_file_path = os.path.join(base_dir, 'app', 'data', 'processed', 'messages_clean.json')
output_file_path = os.path.join(base_dir, 'app', 'data', 'processed', 'merged_data.json')


def merge_conversations_with_messages():
    """
    Charge les fichiers de conversations et de messages, les fusionne
    en se basant sur le contact_id, puis sauvegarde le résultat
    dans un nouveau fichier JSON.
    """
    print("Début de la fusion des conversations et des messages...")

    # 1. Charger les données des fichiers
    try:
        with open(conversations_file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        with open(messages_file_path, 'r', encoding='utf-8') as f:
            messages = json.load(f)

    except FileNotFoundError as e:
        print(f"Erreur : Un des fichiers n'a pas été trouvé. Veuillez vérifier les chemins. Erreur : {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON. Le fichier est-il corrompu ? Erreur : {e}")
        return

    # 2. Ignorer la première ligne qui est une ligne d'en-tête
    conversations_data = conversations[1:]
    messages_data = messages[1:]

    # 3. Créer un dictionnaire pour grouper les messages par contact_id.
    # Ceci est la nouvelle logique de fusion.
    messages_by_contact_id = {}
    for message in messages_data:
        # Nous allons utiliser le sender_id du message comme clé pour le contact.
        # Cela semble être la seule façon de lier les messages aux contacts.
        sender_id = message.get('sender_id')
        if sender_id:
            if sender_id not in messages_by_contact_id:
                messages_by_contact_id[sender_id] = []
            messages_by_contact_id[sender_id].append(message)

    # 4. Fusionner les conversations avec leurs messages correspondants en utilisant contact_id
    merged_conversations = []
    for conversation in conversations_data:
        # Utiliser le contact_id de la conversation pour trouver les messages correspondants
        contact_id = conversation.get('contact_id')
        if contact_id and contact_id in messages_by_contact_id:
            # Ajouter une nouvelle clé 'messages' à l'objet conversation
            conversation['messages'] = messages_by_contact_id[contact_id]
        else:
            conversation['messages'] = []  # Aucun message trouvé pour ce contact_id
        merged_conversations.append(conversation)

    # 5. Sauvegarder le résultat dans un nouveau fichier JSON
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_conversations, f, indent=4, ensure_ascii=False)
        print(f"Fusion réussie ! Le fichier a été sauvegardé ici : {output_file_path}")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'écriture du fichier. Erreur : {e}")


def run():
    """Point d'entrée pour la fusion automatique."""
    merge_conversations_with_messages()


# Exécuter la fonction si le script est lancé directement
if __name__ == '__main__':
    run()