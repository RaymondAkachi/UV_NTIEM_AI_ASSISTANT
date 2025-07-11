import base64
import logging
from together import Together
import os
from uuid import uuid4
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os
from settings import settings

logging.basicConfig(level=logging.INFO)


class TogetherImageGenerator:
    """
    A client for generating images using the Together API asynchronously.

    Attributes:
        api_key (str): The API key for authentication.
        base_url (str): The base URL of the Together API.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(self):
        """
        Initialize the image generator with an API key.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL of the Together API (default: "https://api.together.ai/v1").
            max_concurrency (int): Maximum number of concurrent requests (default: 5).

        Raises:
            ValueError: If the API key is not provided.
        """

        self.api_key = settings.TOGETHERAI_API_KEY
        self.logger = logging.getLogger(__name__)
        self.together_client = Together(api_key=self.api_key)

    def upload_to_s3(self, image_path):
        try:
            s3_client = boto3.client('s3',
                                     aws_access_key_id=settings.S3_BUCKET_ACCESS_KEY_ID,
                                     aws_secret_access_key=settings.S3_BUCKET_SECRET_ACCESS_KEY,
                                     region_name='eu-north-1')

            # file_extension = os.path.splitext(image_path)[1]  # e.g., '.png'
            video_key = f"images/{image_path}"
            s3_client.upload_file(
                image_path,
                "ntiembotbucket",
                video_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            url = f"{settings.S3_BUCKET_URL}/{video_key}"
            os.remove(image_path)
            return url
        except ValueError as e:
            print(e)
        except FileNotFoundError:
            print("Video file not found")
        except NoCredentialsError:
            print("AWS credentials not found. Configure using:")
            print("1. AWS CLI: 'aws configure'")
            print("2. Environment variables: AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY")
        except ClientError as e:
            print(f"AWS Client Error: {e.response['Error']['Message']}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

    def generate_image(self, prompt: str) -> bytes:
        """Generate an image from a prompt using Together AI."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            output_path = f"images/single_image{str(uuid4())}.png"
            self.logger.info(f"Generating image for prompt: '{prompt}'")

            response = self.together_client.images.generate(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell-Free",
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json",
            )

            image_data = base64.b64decode(response.data[0].b64_json)

            with open(output_path, "wb") as f:
                f.write(image_data)
            self.logger.info(f"Image saved to {output_path}")

            url = self.upload_to_s3(output_path)
            return url
        except Exception as e:
            print(e)

# if __name__ == "__main__":
#     x = TogetherImageGenerator().generate_image(
#         "Make me an image of a gorilla that is dancing")
#     print(x)


# async def single_image():
#     generator = TogetherImageGenerator(
#         "49d29ffd4bd6b87e7a652ef93e35c7d90eab9a16b47a0fdabaf5c168e9404eed")
#     # image_data = await generator.generate_image(
#     #     prompt="A cozy cabin in the woods",
#     #     model="black-forest-labs/FLUX.1-schnell-Free",
#     #     width=1024,
#     #     height=768
#     # )
#     image_data = await generator.generate_image(
#         prompt="Generate a realistic picture of a large majestic lion",
#         output_path=f"single_image{str(uuid4())}.png"
#     )
