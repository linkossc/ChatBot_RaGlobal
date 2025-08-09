# Mettez ces imports au début du fichier
from flask import render_template, request, jsonify, Blueprint
from app.models.ChatBot.chatbot import Chatbot

# Le paramètre 'template_folder' est ajouté pour indiquer à Flask
# de chercher le dossier 'templates' deux niveaux au-dessus du fichier actuel.
main_bp = Blueprint('main', __name__, template_folder='../../templates')

# Initialisation du chatbot une seule fois pour éviter de recharger le modèle à chaque requête
try:
    # Le paramètre 'training_data_file' a été retiré.
    chatbot = Chatbot(model_name="logistic_regression")
    print("Chatbot initialisé avec succès pour l'application Flask.")
except Exception as e:
    print(f"Échec de l'initialisation du chatbot : {e}")
    chatbot = None

@main_bp.route('/')
def index():
    """Route pour la page d'accueil (interface du chatbot)."""
    return render_template('index.html')

@main_bp.route('/chatbot_response', methods=['POST'])
def get_chatbot_response():
    """
    Route API pour obtenir une réponse du chatbot.
    Reçoit un message du client et renvoie la réponse prédite.
    """
    if chatbot is None:
        return jsonify({"response": "Le chatbot est en maintenance. Veuillez réessayer plus tard."}), 503

    # Récupérer le message du client depuis la requête JSON
    data = request.get_json()
    client_message = data.get('message', '')

    if not client_message:
        return jsonify({"response": "Message invalide."}), 400

    # Obtenir la réponse du chatbot
    bot_response = chatbot.get_response(client_message)

    # Renvoyer la réponse au format JSON
    return jsonify({"response": bot_response})