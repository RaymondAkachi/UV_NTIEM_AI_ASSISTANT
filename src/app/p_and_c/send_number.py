import asyncio
from qstash import QStash
from app.settings import settings
import logging
from typing import Dict, Any, Optional
import uuid
import hmac
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def schedule_number_send(
    user_number: str,
    text: str,
    additional_data: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Schedule a task to be executed in 2 minutes using QStash, sending user number and text.

    Args:
        destination_url (str): The public API endpoint to trigger (e.g., 'https://your-api.com/task')
        user_number (str): The user's phone number (e.g., '+1234567890')
        text (str): Custom text (e.g., prayer query)
        additional_data (Dict[str, Any], optional): Additional metadata to include in the payload

    Returns:
        bool: True if the task was scheduled successfully, False otherwise
    """
    try:
        # Initialize QStash client
        client = QStash(token=settings.QSTASH_TOKEN)
        destination = "https://stirring-piranha-especially.ngrok-free.app/send-number"
        # Create payload
        payload = {
            "user_number": user_number,
            "text": text,
            "task_id": str(uuid.uuid4()),  # Unique task ID for tracking
            **(additional_data or {})
        }

        # Schedule the task with a 2-minute (120 seconds) delay
        response = await client.publishJSON(
            url=destination,
            body=payload,
            headers={
                "Content-Type": "application/json",
                # Delay execution by 120 seconds (2 minutes)
                "Upstash-Delay": "120s"
            }
        )

        logger.info(
            f"Task scheduled successfully for {destination}. Message ID: {response['messageId']}")
        return True

    except Exception as e:
        logger.error(
            f"Failed to schedule task for {destination}: {str(e)}")
        return False


def verify_qstash_signature(body: bytes, signature: str) -> bool:
    """
    Verify the QStash signature to ensure the request is from QStash.

    Args:
        body (bytes): The raw request body
        signature (str): The Upstash-Signature header

    Returns:
        bool: True if the signature is valid, False otherwise
    """
    try:
        # Compute HMAC using the current signing key
        expected_signature = hmac.new(
            settings.QSTASH_CURRENT_SIGNING_KEY.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()

        if hmac.compare_digest(signature, expected_signature):
            return True

        # Try the next signing key (for key rotation)
        expected_signature = hmac.new(
            settings.QSTASH_NEXT_SIGNING_KEY.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    except Exception as e:
        logger.error(f"Error verifying QStash signature: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage
    async def main():
        # Replace with your FastAPI endpoint
        user_number = "2349094540644"
        text = "I need prayer for my marriage"
        additional_data = {"priority": "high"}

        success = await schedule_number_send(user_number, text, additional_data)
        if success:
            print("Task scheduled to run in 2 minutes.")
        else:
            print("Failed to schedule task.")

    asyncio.run(main())
