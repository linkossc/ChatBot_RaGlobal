# app/__init__.py
from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')

    # Enregistrer les routes
    from app.routes.main import bp as main_bp
    app.register_blueprint(main_bp)

    return app