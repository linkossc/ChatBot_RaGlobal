import os
import pickle
import json
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from app.models.utils import load_data, MODEL_DIR

# Ignorer les avertissements UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("🔹 Entraînement Logistic Regression...")
    df = load_data()
    if df is None or df.empty:
        print("Abandon de l'entraînement car les données n'ont pas pu être chargées.")
        return

    # Créer un dossier indépendant pour ce modèle
    LOGISTIC_REGRESSION_DIR = os.path.join(MODEL_DIR, "logistic_regression")
    os.makedirs(LOGISTIC_REGRESSION_DIR, exist_ok=True)

    # Encoder les labels et sauvegarder l'encodeur
    label_encoder = LabelEncoder()
    df["label_enc"] = label_encoder.fit_transform(df["label"])
    pickle.dump(label_encoder, open(os.path.join(LOGISTIC_REGRESSION_DIR, "label_encoder.pkl"), "wb"))

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_enc"], test_size=0.2, random_state=42
    )

    # Vectorisation TF-IDF et sauvegarde du vectoriseur
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    pickle.dump(vectorizer, open(os.path.join(LOGISTIC_REGRESSION_DIR, "tfidf_vectorizer.pkl"), "wb"))

    # Entraînement du modèle
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    unique_labels = np.unique(y_test)
    target_names_filtered = label_encoder.inverse_transform(unique_labels)
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names_filtered,
                                zero_division=0))

    # Sauvegarder le modèle dans son propre dossier
    model_path = os.path.join(LOGISTIC_REGRESSION_DIR, "logistic_regression.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Sauvegarder les métriques dans le même dossier
    metrics_path = os.path.join(LOGISTIC_REGRESSION_DIR, "metrics_logistic_regression.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }, f, indent=4)

    print("✅ Logistic Regression entraîné et sauvegardé.")
    print(f"📊 Métriques sauvegardées dans {metrics_path}")


if __name__ == "__main__":
    main()