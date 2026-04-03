from flask import Flask, render_template
from .routes import main
from .services.prediction_service import load_pipeline
from config import Config
import logging
import os

# Absolute paths derived from this file's location.
# Works regardless of which directory the app is launched from.
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # credit_card_app/
_REPO_ROOT = os.path.dirname(_APP_ROOT)                                    # GoCreadi/
_PIPELINE_PATH = os.path.join(_REPO_ROOT, "model_artifacts", "pipeline.pkl")
_LOGS_DIR = os.path.join(_APP_ROOT, "logs")


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Load ML pipeline
    app.config["PIPELINE"] = load_pipeline(_PIPELINE_PATH)

    # Configure logging
    if not os.path.exists(_LOGS_DIR):
        os.mkdir(_LOGS_DIR)

    file_handler = logging.FileHandler(os.path.join(_LOGS_DIR, "app.log"))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    app.register_blueprint(main)
    
    #Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("error.html", message="Page not found."), 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Server Error: {error}")
        return render_template("error.html", message="Internal server error. Please try again."), 500

    return app