from flask import Flask
from flask_cors import CORS
import os


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
    CORS(app)

    from app.routes.main import main_bp
    from app.routes.upload import upload_bp
    from app.routes.chat import chat_bp
    from app.routes.summary import summary_bp
    from app.routes.explain import explain_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp, url_prefix="/upload")
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(summary_bp, url_prefix="/summary")
    app.register_blueprint(explain_bp, url_prefix="/explain")

    return app
