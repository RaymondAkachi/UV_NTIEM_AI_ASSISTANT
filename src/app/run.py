import uvicorn
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the FastAPI application using Uvicorn."""
    # Load non-sensitive environment variables from .env file (optional)
    if os.path.exists('.env'):
        load_dotenv('.env')
        logger.info("Loaded .env file for non-sensitive configurations")
    else:
        logger.info(
            "No .env file found; relying on runtime environment variables")

    # Validate required sensitive environment variables (uncomment if needed)
    """
    required_sensitive_vars = ["OPENAI_API_KEY", "WHATSAPP_TOKEN", "ASSEMBLYAI_API_KEY"]
    missing_vars = [var for var in required_sensitive_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required sensitive environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required sensitive environment variables: {', '.join(missing_vars)}")
    """

    # Uvicorn configuration
    host = "127.0.0.1"  # Bind to all interfaces
    port = 8000       # Default port
    reload = False    # Disable reload by default (enable for development)

    logger.info(
        f"Starting Uvicorn server on {host}:{port}, reload={reload}")

    # Run Uvicorn server with import string
    uvicorn.run(
        "main:app",  # Import string for the FastAPI app
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        # Handle long-lived WebSocket connections (e.g., WhatsApp webhooks)
        timeout_keep_alive=30,
        use_colors=True
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
