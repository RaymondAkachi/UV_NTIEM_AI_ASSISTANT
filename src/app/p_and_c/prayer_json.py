import json
import boto3
import logging
from typing import Dict, Any
import os
from botocore.exceptions import ClientError
from app.settings import settings
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default prayer feedback data (embedded to avoid file dependency)
DEFAULT_PRAYER_DATA = {
    "prayer_info": {
        "Marriage": {
            "prayer": "verses: 1 Corinthians 13:4-7, Ephesians 5:25-33, Ecclesiastes 4:12\n\nprayer: Heavenly Father, bless this marriage with Your love, patience, and unity. Strengthen the bond between husband and wife, guiding them to honor and cherish each other as You intend. May their union be a reflection of Your grace and a testimony of Your faithfulness. Amen.",
            "contact": "Church Marriage Department: +44 7368 877315"
        },
        "Career": {
            "prayer": "verses: Colossians 3:23-24, Proverbs 16:3, Psalm 90:17\n\nprayer: Lord, guide me in my career and grant me wisdom to excel in my work. Open doors of opportunity and establish the work of my hands. Help me to serve with integrity and glorify You in all I do. Amen.",
            "contact": "Church Career Prayerline: +234 803 223 5209"
        },
        "Finances": {
            "prayer": "verses: Philippians 4:19, Malachi 3:10, Deuteronomy 8:18\n\nprayer: God of provision, I trust You to meet all my financial needs according to Your riches. Grant me wisdom to steward resources faithfully and open windows of blessing to overflow in my life. Remove anxiety and lead me to prosperity for Your glory. Amen.",
            "contact": "Church Finances Prayerline: +234 803 223 5209"
        },
        "Health": {
            "prayer": "verses: Jeremiah 30:17, Psalm 30:2, 3 John 1:2\n\nprayer: Heavenly Father, I seek Your healing touch for my body, mind, and soul. Restore my health, grant strength to my frame, and guide those providing care. May Your peace and wholeness envelop me, in Jesus’ name. Amen.",
            "contact": "Church Health Prayerline. Call: +234 803 319 4594"
        },
        "Children": {
            "prayer": "verses: Proverbs 22:6, Psalm 127:3-5, Isaiah 54:13\n\nprayer: Lord, I lift up my children to You, asking for Your protection and guidance. Train them in Your ways, fill them with wisdom, and let them grow in faith and favor. Surround them with Your love and keep them safe. Amen.",
            "contact": "Church Children Prayerline: +234 803 223 5209"
        },
        "Direction": {
            "prayer": "verses: Proverbs 3:5-6, Psalm 32:8, James 1:5\n\nprayer: Father, I seek Your guidance for my path. Direct my steps, grant me clarity, and fill me with Your wisdom. Let Your will be my guide, and may I walk confidently in Your purpose for my life. Amen.",
            "contact": "Church Direction Prayerline: +234 803 223 5209"
        },
        "Spiritual_Attack": {
            "prayer": "verses: Ephesians 6:12, Psalm 91:1-4, James 4:7\n\nprayer: Almighty God, I stand firm in Your strength against all spiritual attacks. Clothe me with Your armor, protect me from evil, and grant me victory through the power of Jesus’ name. Let Your peace guard my heart. Amen.",
            "contact": "Focus on the prayer: +234 803 223 5209"
        },
        "Others": {
            "prayer": "verses: Philippians 4:6-7, 1 Thessalonians 5:17, Matthew 7:7\n\nprayer: Loving Father, I bring my unspoken needs before You, trusting in Your infinite wisdom. Hear my heart’s cry, grant me peace, and answer according to Your perfect will. Help me to trust and pray continually. Amen.",
            "contact": "Prayer Helpline: +234 803 223 5209"
        }
    },
    "prayer_details": "Please select out of these options what exactly you would like prayer for.\n\n\n1. Marriage\n\n2. Career\n\n3. Finances\n\n4. Health\n\n5. Children\n\n6. Direction\n\n7. Spiritual Attack\n\n8. Others",
    "prayer_list": {"1": "Marriage", "2": "Career", "3": "Finances", "4": "Health", "5": "Children", "6": "Direction", "8": "Others", "7": "Spiritual Attack"},
    "counsellor_number": "2349094540644",
    "counselling_info": {
        "Marriage": "verses: Ephesians 5:25-33, 1 Corinthians 13:4-7, Ecclesiastes 4:12\n\ncounselling: Build your marriage on love, respect, and mutual support. Communicate openly and seek God’s guidance to strengthen your bond. A cord of three strands—husband, wife, and God—is not easily broken.",
        "Career": "verses: Colossians 3:23-24, Proverbs 16:3, Psalm 90:17\n\ncounselling: Work diligently as unto the Lord, trusting Him to guide your career path. Commit your plans to God, and He will establish your steps with purpose and success.",
        "Finances": "verses: Philippians 4:19, Malachi 3:10, Deuteronomy 8:18\n\ncounselling: Trust God to provide for your financial needs. Practice wise stewardship, give generously, and seek His kingdom first, knowing He will supply all you need.",
        "Health": "verses: Jeremiah 30:17, Psalm 30:2, 3 John 1:2\n\ncounselling: Seek God’s healing and strength for your body and mind. Rest in His care, follow wise medical advice, and trust Him to restore your health and vitality.",
        "Children": "verses: Proverbs 22:6, Psalm 127:3-5, Isaiah 54:13\n\ncounselling: Raise your children with love and godly instruction. Trust God to protect and guide them, knowing they are a precious gift and heritage from Him.",
        "Direction": "verses: Proverbs 3:5-6, Psalm 32:8, James 1:5\n\ncounselling: Seek God’s wisdom in every decision. Trust Him to guide your path, and He will direct you with clarity and purpose as you lean on His understanding.",
        "Spiritual_Attack": "verses: Ephesians 6:12, Psalm 91:1-4, James 4:7\n\ncounselling: Stand firm in God’s strength against spiritual challenges. Put on His armor, resist the enemy with faith, and rest under His protective wings.",
        "Others": "verses: Philippians 4:6-7, 1 Thessalonians 5:17, Matthew 7:7\n\ncounselling: Bring all your concerns to God in prayer. Trust His wisdom and timing, and He will grant you peace and answers as you seek Him earnestly."
    }
}


def create_and_upload_prayer_feedback(bucket_name: str, file_key: str) -> bool:
    """
    Create a JSON file from the data in initial_prayer.json (or default data) and upload it to an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket
        file_key (str): S3 key (path) for the JSON file (e.g., 'prayer_feedback.json')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_BUCKET_SECRET_ACCESS_KEY
        )

        # Try to read from initial_prayer.json, use default if not found
        prayer_data = DEFAULT_PRAYER_DATA
        if os.path.exists("initial_prayer.json"):
            with open("initial_prayer.json", 'r', encoding='utf-8') as f:
                prayer_data = json.load(f)
                logger.info("Loaded prayer data from initial_prayer.json")
        else:
            logger.warning(
                "initial_prayer.json not found, using default prayer data")

        # Convert prayer data to JSON string
        json_data = json.dumps(prayer_data, indent=2)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )

        logger.info(
            f"Successfully created and uploaded {file_key} to S3 bucket {bucket_name}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse initial_prayer.json: {str(e)}")
        return False
    except ClientError as e:
        logger.error(
            f"Failed to upload {file_key} to S3 bucket {bucket_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error while creating/uploading {file_key}: {str(e)}")
        return False


def modify_prayer_feedback(bucket_name: str, file_key: str, updates: Dict[str, Any]) -> bool:
    """
    Modify the prayer feedback JSON file in S3 with provided updates.

    Args:
        bucket_name (str): Name of the S3 bucket
        file_key (str): S3 key (path) for the JSON file (e.g., 'prayer_feedback.json')
        updates (Dict[str, Any]): Dictionary with updates to apply (e.g., new category, updated verses)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_BUCKET_SECRET_ACCESS_KEY
        )

        # Download existing JSON file
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        current_data = json.loads(response['Body'].read().decode('utf-8'))

        # Apply updates
        current_data.update(updates)

        # Convert updated data to JSON string
        updated_json = json.dumps(current_data, indent=2)

        # Upload updated file to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=updated_json.encode('utf-8'),
            ContentType='application/json'
        )

        logger.info(
            f"Successfully updated {file_key} in S3 bucket {bucket_name}")
        return True

    except ClientError as e:
        logger.error(
            f"Failed to update {file_key} in S3 bucket {bucket_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while updating {file_key}: {str(e)}")
        return False


def replace_prayer_feedback(bucket_name: str, file_key: str, new_data: Union[Dict[str, Any], str]) -> bool:
    """
    Replace the entire prayer_feedback.json file in the S3 bucket with new JSON content.

    Args:
        bucket_name (str): Name of the S3 bucket (e.g., 'your-bucket-name').
        file_key (str): S3 key (path) for the JSON file (e.g., 'prayer_feedback.json').
        new_data (Union[Dict[str, Any], str]): New JSON content as a dictionary or JSON string.

    Returns:
        bool: True if the replacement was successful, False otherwise.
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.S3_BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=settings.S3_BUCKET_SECRET_ACCESS_KEY
        )

        # Convert new_data to JSON string if it's a dictionary
        if isinstance(new_data, dict):
            json_data = json.dumps(new_data, indent=2)
        else:
            # Validate JSON string
            json.loads(new_data)  # Raises JSONDecodeError if invalid
            json_data = new_data

        # Upload new JSON content to S3, overwriting the existing file
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )

        logger.info(
            f"Successfully replaced {file_key} in S3 bucket {bucket_name}")
        return True

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON provided: {str(e)}")
        return False
    except ClientError as e:
        logger.error(
            f"Failed to replace {file_key} in S3 bucket {bucket_name}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while replacing {file_key}: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage
    BUCKET_NAME = settings.S3_BUCKET_NAME  # Ensure this is set in app.settings
    FILE_KEY = "prayer_feedback.json"

    # Create and upload the initial JSON file
    # create_and_upload_prayer_feedback(BUCKET_NAME, FILE_KEY)
    replace_prayer_feedback(BUCKET_NAME, FILE_KEY, DEFAULT_PRAYER_DATA)
