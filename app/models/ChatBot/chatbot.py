import os
import pickle
import sys
import numpy as np
import warnings
import json
import random

# Ignorer les avertissements pour garder la console propre
warnings.filterwarnings("ignore")

# Définir les chemins de base pour les modèles et les données
# Le chemin est relatif à l'emplacement de ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correction : le dossier "saved" est un dossier parent du dossier du script
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "saved")

# Le chemin vers le dataset est un niveau encore plus haut, dans 'data/training'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data", "training")

# Mise à jour du nom du fichier pour correspondre à votre fichier 'synthetic_conversations.json'
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "synthetic_conversations.json")

class Chatbot:
    """
    Une classe simple pour un chatbot qui utilise un modèle de classification
    pour prédire un statut et générer une réponse à partir d'un dataset existant.
    """

    def __init__(self, model_name="logistic_regression"):
        """
        Initialise le chatbot en chargeant le modèle de classification,
        le vectorizer et le dataset de formation.

        Args:
            model_name (str): Le nom du modèle de classification à utiliser.
        """
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.training_data = None

        self._load_resources()

    def _load_resources(self):
        """Charge le modèle, le vectorizer, le LabelEncoder et le dataset."""
        print("🤖 Initialisation du Chatbot...")
        # Le nom du modèle à charger est maintenant "logistic_regression"
        model_path_dir = os.path.join(MODEL_DIR, self.model_name)

        if not os.path.exists(model_path_dir):
            raise FileNotFoundError(f"Dossier du modèle introuvable : {model_path_dir}")

        try:
            # Chargement du LabelEncoder
            with open(os.path.join(model_path_dir, "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)

            # Chargement du TF-IDF vectorizer
            with open(os.path.join(model_path_dir, "tfidf_vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)

            # Chargement du modèle
            model_file = f"{self.model_name}.pkl"
            with open(os.path.join(model_path_dir, model_file), "rb") as f:
                self.model = pickle.load(f)

            # Chargement du dataset de formation
            with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)

            print("✅ Ressources chargées avec succès.")
        except FileNotFoundError as e:
            print(f"❌ Erreur lors du chargement des ressources : {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Une erreur inattendue est survenue lors du chargement : {e}")
            sys.exit(1)

    def classify_intent(self, user_message):
        """
        Classe l'intention du message utilisateur en utilisant le modèle.

        Args:
            user_message (str): Le message du client.

        Returns:
            str: Le statut (l'intention) prédit.
        """
        if self.model and self.vectorizer:
            user_message_vectorized = self.vectorizer.transform([user_message])
            pred_label_index = self.model.predict(user_message_vectorized)
            return self.label_encoder.inverse_transform(pred_label_index)[0]
        else:
            return "unknown"

    def get_response(self, user_message):
        """
        Génère une réponse basée sur l'intention du message utilisateur.

        Args:
            user_message (str): Le message du client.

        Returns:
            str: Une réponse générée ou un message d'erreur.
        """
        # 1. Classifier l'intention du client
        intent = self.classify_intent(user_message)
        print(f"Prédiction de l'intention du client : '{intent}'")

        if intent == "unknown":
            return "Je ne suis pas sûr de ce que vous voulez dire. Pouvez-vous reformuler ?"

        # 2. Chercher des réponses correspondantes dans le dataset
        possible_responses = []
        for conversation in self.training_data:
            if conversation.get('status') == intent:
                # Chercher la première réponse de type 'user' ou 'echo' dans la conversation
                for message in conversation.get('messages', []):
                    if message.get('sender_type') in ['user', 'echo']:
                        if message.get('text'):
                            possible_responses.append(message.get('text'))
                        # On ne prend qu'une seule réponse par conversation pour éviter les doublons
                        break

        # 3. Retourner une réponse aléatoire si des options sont trouvées
        # Correction d'indentation : cette ligne doit être alignée avec le 'if' ci-dessus
        if possible_responses:
            return random.choice(possible_responses)
        else:
            return "Je n'ai pas de réponse correspondante pour cette intention."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py '<message_du_client>'")
        sys.exit(1)

    client_message = sys.argv[1]

    try:
        # Création et utilisation du chatbot
        # Le nom du modèle est maintenant "logistic_regression"
        my_chatbot = Chatbot(model_name="logistic_regression")
        response = my_chatbot.get_response(client_message)
        print("\nRéponse du Chatbot :")
        print(response)
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        sys.exit(1)