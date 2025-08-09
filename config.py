import os

# Racine du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossiers données
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DATA_TRAINING = os.path.join(BASE_DIR, 'data', 'training')

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_TRAINING, exist_ok=True)

CONTACTS_CSV = os.path.join(DATA_RAW, "contacts.csv")
CONVERSATIONS_CSV = os.path.join(DATA_RAW, "conversations-csv.csv")
MESSAGES_CSV = os.path.join(DATA_RAW, "messages-csv.csv")

# Flags auto nettoyage / fusion
AUTO_CLEAN_DATA = False   # Si True, nettoie les CSV au démarrage
AUTO_MERGE_DATA = False   # Si True, merge les JSON nettoyés au démarrage
AUTO_PREPARE_TRAINING_DATASET = False # Si True, prépare le dataset de formation
AUTO_GENERATE_SYNTHETIC_DATA = False # Nouveau flag pour la génération de données synthétiques
AUTO_AUGMENT_SYNTHETIC_DATA = False # Nouveau flag pour l'enrichissement
AUTO_CLEAN_TRAINING_DATA = False # Si True, nettoie le training_dataset.json

# Activation ou désactivation de l'entraînement automatique au démarrage
AUTO_TRAIN_RANDOM_FOREST = False
AUTO_TRAIN_NAIVE_BAYES = False
AUTO_TRAIN_LOGISTIC_REGRESSION = False
AUTO_TRAIN_LSTM = False