from dotenv import load_dotenv
load_dotenv()

from app import create_app
import os

app = create_app()

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
