import boto3
import time
# import asyncio
import os
from botocore.exceptions import ClientError, NoCredentialsError
from settings import settings


class VoiceCreation:
    def __init__(self, text):
        self.polly_client = boto3.client(
            'polly',
            aws_access_key_id=settings.POLLY_ACCESS_KEY_ID,  # From the IAM CSV
            aws_secret_access_key=settings.POLLY_SECRET_ACCESS_KEY,  # From the IAM CSV
            # Choose a region close to you, e.g., 'us-east-1' for N. Virginia
            region_name=settings.POLLY_REGION_NAME
        )
        self.text = text

    def upload_to_s3(self, output_path):
        try:
            s3_client = boto3.client('s3',
                                     aws_access_key_id=settings.S3_BUCKET_ACCESS_KEY_ID,
                                     aws_secret_access_key=settings.S3_BUCKET_SECRET_ACCESS_KEY,
                                     region_name='eu-north-1')

            # file_extension = os.path.splitext(image_path)[1]  # e.g., '.png'
            video_key = f"audio/{output_path}"
            s3_client.upload_file(
                output_path,
                "ntiembotbucket",
                video_key,
                ExtraArgs={'ContentType': 'audio/mpeg'}
            )
            url = f"{settings.S3_BUCKET_URL}/{video_key}"
            os.remove(output_path)
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

    def text_to_speech(self):
        # Call Polly to synthesize speech
        try:
            response = self.polly_client.synthesize_speech(
                Text=self.text,
                OutputFormat='mp3',  # WhatsApp-compatible format
                # A natural-sounding voice (see AWS docs for other options)
                VoiceId='Joanna'
            )

            # Save the audio to a file
            # output_file = f"audio/response_{int(time.time())}.mp3"
            output_file = f"voices/response_{int(time.time())}.mp3"
            with open(output_file, 'wb') as out:
                out.write(response['AudioStream'].read())

            url = self.upload_to_s3(output_file)
            return url
        except Exception as e:
            print(e)


# Initialize the Polly client with your credentials
if __name__ == "__main__":
    x = VoiceCreation(
        "I am the fastest man alive right now").text_to_speech()
    print(x)


# async def hello():
#     print("Hello")
# asyncio.run(text_to_speech(
#     "Udochi Raymond is the fastst runner in Pacemark school"))
