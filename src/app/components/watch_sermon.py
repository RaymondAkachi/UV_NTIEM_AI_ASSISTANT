import json
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import boto3
import csv
from io import StringIO
from botocore.exceptions import ClientError
from app.settings import settings
import asyncio  # For testing purposes
import time  # For in-script testing
from dotenv import load_dotenv
import os
from app.settings import settings

load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))


S3_SECRET_ACCESS_KEY = settings.S3_BUCKET_SECRET_ACCESS_KEY
S3_ACCESS_KEY = settings.S3_BUCKET_ACCESS_KEY_ID
S3_BUCKET_REGION = settings.S3_REGION_NAME


def find_video_links(search_terms: dict,  csv_key='videos.csv'):
    """
    Searches CSV in S3 bucket for matching title or upload time
    Returns list of dictionaries with s3bucketlink and socialLink
    """
    s3 = boto3.client(
        's3',
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=S3_BUCKET_REGION)

    bucket_name = settings.S3_BUCKET_NAME  # e.g., 'ntiembotbucket'
    try:
        # Get CSV file from S3
        csv_obj = s3.get_object(Bucket=bucket_name, Key=csv_key)
        csv_content = csv_obj['Body'].read().decode('utf-8')

        # Parse CSV content
        csv_buffer = StringIO(csv_content)
        reader = csv.DictReader(csv_buffer)
        rows = list(reader)

        results = []
        modified_title = str(search_terms['title']).strip().replace(
            ' ', '_').lower()

        # First search by Title
        if search_terms['title']:
            requested_title = search_terms['title']
            for row in rows:
                if row['Title'].lower().split('.')[0] == modified_title:
                    results.append({
                        's3VideoLink': row['s3VideoLink'],
                        'socialVideoLink': row['socialVideoLink'],
                        'match_type': 'title',
                        'title': requested_title
                    })
                    # found_in_titles = True

        if search_terms['date']:
            requested_date = search_terms['date']
            for row in rows:
                if row['UploadTime'] == search_terms['date']:
                    results.append({
                        's3VideoLink': row['s3VideoLink'],
                        'socialVideoLink': row['socialVideoLink'],
                        'match_type': 'upload_time',
                        'upload_time': requested_date
                    })
        if results == []:
            return f"Sorry we could not find any video for the values given"

        return results

    except ClientError as e:
        print(f"S3 Error: {e.response['Error']['Message']}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


class GetSermon:
    def __init__(self, sermon_info: str) -> None:
        self.sermon_info = sermon_info
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["request"],
            template="""You are an assistant tasked with extracting sermon titles and dates from user requests. Extract the sermon title and the date it was preached, returning them in a JSON object with "title" and "date" keys. Use `null` for any missing or unclear values.

- **Title Extraction**: Identify the title from keywords like "titled", "called", "named", or direct references (e.g., "the sermon Hope"). Remove any quotation marks from the title in the output (e.g., "Hope" becomes Hope) and preserve its original casing and formatting.
- **Date Extraction**: Identify the date from phrases like "preached on", "delivered on", "from", or direct date mentions (e.g., "October 12, 2024", "2024-12-23", "12/12/2024"). Convert all dates to the format "day month year", where:
  - Day is ordinal (e.g., "12th", "23rd").
  - Month is the full name (e.g., "October", "December").
  - Year is four digits (e.g., "2024").
  Examples:
  - "2024-12-23" → "23rd December 2024"
  - "12/12/2024" → "12th December 2024"
  - "October 12, 2024" → "12th October 2024"

Be flexible and case-insensitive with phrasing. Interpret ambiguous inputs logically, using contextual clues (e.g., "last Sunday" could imply a recent date if context is provided, otherwise use `null`). For incomplete dates (e.g., "October 2024"), return `null` unless the day is inferable.

**Output**: A JSON object like {{ "title": "Hope", "date": "12th October 2024" }}.

**Examples**:
- "Sermon titled Hope preached on 2024-12-23" → {{ "title": "Hope", "date": "23rd December 2024" }}
- "Get the sermon 'Grace' from 12/12/2024" → {{ "title": "Grace", "date": "12th December 2024" }}
- "Sermon on October 12, 2024" → {{ "title": null, "date": "12th October 2024" }}
- "Sermon called Faith" → {{ "title": "Faith", "date": null }}
- "Latest sermon" → {{ "title": null, "date": null }}
- "Get me the sermon the grace of God on 12th October 2024" -> {{"title": "the grace of God", "date": "12th October 2024"}}
- 'Get me the sermon on the peace of God' -> {{'title': "the peace of God", "date": "null"}}

Extract the title and date from: {request}"""
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser(
        ) | RunnableLambda(self._parse_json)

    def _parse_json(self, output: str) -> dict:
        """Parse the output string into a JSON object, with fallback for errors."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"title": None, "date": None}

    async def result_getter(self) -> bool:
        """Get's result from the chain and saves it in results"""
        # self.chain = self._create_chain
        results = await self.chain.ainvoke({"request": self.sermon_info})
        if results['title'] or results['date']:
            return results
        else:
            return None

    async def get_sermons(self):
        sermon_info = await self.result_getter()
        if sermon_info:
            results = find_video_links(search_terms=sermon_info)
            return results
        else:
            "Please mention a date or time that the sermon was preached"


if __name__ == "__main__":
    a = time.time()
    y = asyncio.run(
        GetSermon("Get me the preaching test_video2").get_sermons())
    print(y)
    b = time.time()
    print(b-a)
