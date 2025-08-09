from flask import Flask
from config import (
    AUTO_CLEAN_DATA,
    AUTO_MERGE_DATA,
    AUTO_PREPARE_TRAINING_DATASET,
    AUTO_GENERATE_SYNTHETIC_DATA,
    AUTO_AUGMENT_SYNTHETIC_DATA,
    AUTO_CLEAN_TRAINING_DATA,
    AUTO_TRAIN_RANDOM_FOREST,
    AUTO_TRAIN_NAIVE_BAYES,
    AUTO_TRAIN_LOGISTIC_REGRESSION,
    AUTO_TRAIN_LSTM
)


# Fonctions d'entraînement des modèles, déplacées depuis app/models/__init__.py
def train_random_forest():
    """
    Lance le script d'entraînement pour le modèle Random Forest.
    """
    try:
        from app.models.train_random_forest import main as rf_train
        rf_train()
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du Random Forest : {e}")

def train_naive_bayes():
    """
    Lance le script d'entraînement pour le modèle Naive Bayes.
    """
    try:
        from app.models.train_naive_bayes import main as nb_train
        nb_train()
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du Naive Bayes : {e}")

def train_logistic_regression():
    """
    Lance le script d'entraînement pour le modèle de Régression Logistique.
    """
    try:
        from app.models.train_logistic_regression import main as lr_train
        lr_train()
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement de la Régression Logistique : {e}")

def train_lstm():
    """
    Lance le script d'entraînement pour le modèle LSTM.
    """
    try:
        from app.models.train_lstm import main as lstm_train
        lstm_train()
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du LSTM : {e}")


def run_training_if_enabled():
    """
    Lance l'entraînement des modèles si les flags correspondants
    sont activés dans config.py.
    """
    if AUTO_TRAIN_RANDOM_FOREST:
        print("Lancement de l'entraînement du modèle Random Forest...")
        train_random_forest()

    if AUTO_TRAIN_NAIVE_BAYES:
        print("Lancement de l'entraînement du modèle Naive Bayes...")
        train_naive_bayes()

    if AUTO_TRAIN_LOGISTIC_REGRESSION:
        print("Lancement de l'entraînement du modèle de Régression Logistique...")
        train_logistic_regression()

    if AUTO_TRAIN_LSTM:
        print("Lancement de l'entraînement du modèle LSTM...")
        train_lstm()


def create_app():
    app = Flask(__name__)
    app.config.from_object('config')

    with app.app_context():
        if AUTO_CLEAN_DATA:
            try:
                from app.services.clean_contacts import clean_contacts
                from app.services.clean_conversations import clean_conversations
                from app.services.clean_messages import clean_messages

                print("🧹 Nettoyage automatique des CSV vers JSON...")
                clean_contacts()
                clean_conversations()
                clean_messages()
                print("✅ Nettoyage terminé.")
            except Exception as e:
                print(f"❌ Erreur nettoyage automatique : {e}")

        if AUTO_MERGE_DATA:
            try:
                from app.services import merge_all
                print("🔗 Fusion automatique des JSON...")
                merge_all.run()
                print("✅ Fusion terminée.")
            except Exception as e:
                print(f"❌ Erreur fusion automatique : {e}")

        if AUTO_PREPARE_TRAINING_DATASET:
            try:
                from app.services import prepare_training_dataset
                print("🔬 Préparation du dataset de formation...")
                prepare_training_dataset.run()
                print("✅ Préparation terminée.")
            except Exception as e:
                print(f"❌ Erreur préparation dataset : {e}")

        # Vous pouvez utiliser ce flag pour la génération initiale
        if AUTO_GENERATE_SYNTHETIC_DATA:
            try:
                from app.services import generate_synthetic_data
                print("🤖 Génération de conversations synthétiques initiales...")
                generate_synthetic_data.run()
                print("✅ Génération initiale terminée.")
            except Exception as e:
                print(f"❌ Erreur génération de données synthétiques : {e}")

        # Nouveau bloc pour l'enrichissement du dataset
        if AUTO_AUGMENT_SYNTHETIC_DATA:
            try:
                from app.services import augment_synthetic_data
                print("📈 Enrichissement du dataset de conversations synthétiques...")
                augment_synthetic_data.run()
                print("✅ Enrichissement terminé.")
            except Exception as e:
                print(f"❌ Erreur enrichissement du dataset : {e}")

        # Nouveau bloc pour le nettoyage du dataset de formation
        if AUTO_CLEAN_TRAINING_DATA:
            try:
                from app.services import clean_training_data
                print("✨ Nettoyage automatique du dataset de formation...")
                clean_training_data.run()
                print("✅ Nettoyage du dataset terminé.")
            except Exception as e:
                print(f"❌ Erreur lors du nettoyage du dataset : {e}")

        # Lancement de l'entraînement des modèles si les flags sont activés
        try:
            print("🧠 Vérification et lancement de l'entraînement des modèles...")
            run_training_if_enabled()
            print("✅ Processus d'entraînement terminé.")
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement des modèles : {e}")

    # C'est la ligne qui a été corrigée.
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    return app
