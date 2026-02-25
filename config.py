import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # General
    PROJECT_NAME = os.getenv("PROJECT_NAME", "Construction Safety AI")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    PROCESSING_DIR = os.path.join(DATA_DIR, "processing")
    VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
    REPORTS_DIR = os.path.join(DATA_DIR, "reports")

    # AWS
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

    # Roboflow
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
    ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION")

    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "osha_regulations")
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    # Slack
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
    ALERT_EMAIL_DEST = os.getenv("ALERT_EMAIL_DEST")

    @classmethod
    def ensure_dirs(cls):
        """Ensure all data directories exist."""
        for path in [cls.INPUT_DIR, cls.PROCESSING_DIR, cls.VECTOR_DB_DIR, cls.REPORTS_DIR]:
            os.makedirs(path, exist_ok=True)

# Create directories on import
Config.ensure_dirs()
