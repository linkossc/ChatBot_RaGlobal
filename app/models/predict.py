import os
import pickle
import sys
import numpy as np
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore")

# Chemin de base pour tous les modèles sauvegardés
# Correction : Le chemin est maintenant relatif à l'emplacement du script predict.py
# pour garantir que le modèle est toujours trouvé, peu importe le répertoire d'exécution.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")


def predict(model_name, new_text):
    """
    Fait une prédiction en utilisant un modèle spécifié.

    Args:
        model_name (str): Le nom du modèle à utiliser ("random_forest", "naive_bayes", "logistic_regression").
        new_text (str): Le texte d'entrée pour la prédiction.

    Returns:
        str: Le label prédit.
    """
    # Définir le chemin d'accès au dossier spécifique du modèle
    model_path_dir = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path_dir):
        raise FileNotFoundError(f"Dossier du modèle introuvable : {model_path_dir}")

    # Charger le LabelEncoder depuis le dossier spécifique au modèle
    with open(os.path.join(model_path_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    if model_name in ["random_forest", "naive_bayes", "logistic_regression"]:
        # Charger TF-IDF vectorizer depuis le dossier spécifique au modèle
        with open(os.path.join(model_path_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        X_vec = vectorizer.transform([new_text])

        # Charger le modèle en utilisant le nom par défaut
        model_file = f"{model_name}.pkl"

        with open(os.path.join(model_path_dir, model_file), "rb") as f:
            model = pickle.load(f)

        pred = model.predict(X_vec)
        return label_encoder.inverse_transform(pred)[0]
    else:
        raise ValueError("Nom de modèle invalide.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_name> '<text>'")
        sys.exit(1)

    model = sys.argv[1]
    text = sys.argv[2]
    try:
        prediction = predict(model, text)
        print(f"Prédiction avec {model} → {prediction}")
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Erreur: {e}")
        sys.exit(1)