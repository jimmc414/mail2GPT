
import boto3
from botocore.client import Config
import os

DO_ACCESS_KEY = os.getenv("DO_ACCESS_KEY")
DO_SECRET_KEY = os.getenv("DO_SECRET_KEY")
DO_BUCKET_NAME = os.getenv("DO_BUCKET_NAME", "your-space-name")
DO_REGION = "nyc3"
DO_ENDPOINT = f"https://{DO_REGION}.digitaloceanspaces.com"

session = boto3.session.Session()

s3_client = session.client(
    "s3",
    region_name=DO_REGION,
    endpoint_url=DO_ENDPOINT,
    aws_access_key_id=DO_ACCESS_KEY,
    aws_secret_access_key=DO_SECRET_KEY,
)

def upload_to_spaces(file_name: str, file_content: bytes) -> str:
    s3_client.put_object(
        Bucket=DO_BUCKET_NAME, 
        Key=file_name, 
        Body=file_content,
        ACL='private'
    )
    return f"{DO_ENDPOINT}/{DO_BUCKET_NAME}/{file_name}"
