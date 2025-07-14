import httpx
from app.settings import settings
import logging

WHATSAPP_API_AUTHORIZATION = settings.WHATSAPP_API_AUTHORIZATION
WHATSAPP_API_URL = settings.WHATSAPP_API_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def message_recieved(number):
    url = WHATSAPP_API_URL
    headers = {
        "Authorization": WHATSAPP_API_AUTHORIZATION,
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": str(number),
        "type": "text",
        "text": {"body": "Hello we have recieved your message, please wait while we process it!"}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url=url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise


async def text_response(number, text):
    url = WHATSAPP_API_URL
    headers = {
        "Authorization": WHATSAPP_API_AUTHORIZATION,
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": str(number),
        "type": "text",
        "text": {"body": str(text)}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url=url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise


async def audio_response(number, audio_url):
    api_url = WHATSAPP_API_URL
    headers = {
        "Authorization": WHATSAPP_API_AUTHORIZATION,
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": str(number),
        "type": "audio",
        "audio": {
            "link": str(audio_url)
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url=api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error sending audio: {str(e)}")
            raise


async def video_response(number, video_url, caption):
    api_url = WHATSAPP_API_URL
    headers = {
        "Authorization": WHATSAPP_API_AUTHORIZATION,
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": str(number),
        "type": "video",
        "video": {
            "link": str(video_url),
            "caption": str(caption)
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url=api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error sending video: {str(e)}")
            raise


async def image_response(number, image_url):
    caption = "Here is your image"
    api_url = WHATSAPP_API_URL
    headers = {
        "Authorization": WHATSAPP_API_AUTHORIZATION,
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": str(number),
        "type": "image",
        "image": {
            "link": str(image_url)
        }
    }
    if caption:
        payload["image"]["caption"] = str(caption)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url=api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error sending image: {str(e)}")
            raise
