#!/usr/bin/env python3
"""
S3 Explorer - A tool to locate and explore HuggingFace models in S3 buckets
"""

import os
import sys
import boto3
import argparse
import logging
from urllib.parse import urlparse
from pprint import pprint

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def explore_s3_path(s3_path, recursive=False, max_depth=2, max_objects=20):
    """
    Explore an S3 path to find model files

    Args:
        s3_path: S3 URI (s3://bucket/path)
        recursive: Whether to explore subdirectories
        max_depth: Maximum depth for recursive exploration
        max_objects: Maximum number of objects to list per directory
    """
    try:
        # Parse the S3 URI
        parsed_url = urlparse(s3_path)
        if not parsed_url.netloc:
            logger.error(f"Invalid S3 URI: {s3_path}")
            return

        bucket_name = parsed_url.netloc
        prefix = parsed_url.path.lstrip("/")

        # Initialize S3 client
        s3_client = boto3.client("s3")
        logger.info(f"Exploring S3 bucket: {bucket_name}, prefix: {prefix}")

        # Function to list objects at a specific prefix
        def list_objects(prefix, current_depth=0):
            try:
                # Ensure prefix ends with a slash if it's not empty
                if prefix and not prefix.endswith("/"):
                    prefix = prefix + "/"

                # List objects
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name, Prefix=prefix, Delimiter="/"
                )

                # Print common prefixes (directories)
                if "CommonPrefixes" in response:
                    dirs = [p["Prefix"] for p in response["CommonPrefixes"]]
                    logger.info(f"Found {len(dirs)} directories at {prefix}")
                    for dir_prefix in dirs[:max_objects]:
                        dir_name = dir_prefix[len(prefix) :].rstrip("/")
                        logger.info(f"Directory: {dir_name}/")

                        # Recursively explore if needed
                        if recursive and current_depth < max_depth:
                            logger.info(f"Exploring: {dir_prefix}")
                            list_objects(dir_prefix, current_depth + 1)
                else:
                    logger.info(f"No subdirectories found at {prefix}")

                # Print objects (files)
                if "Contents" in response:
                    files = [
                        obj
                        for obj in response["Contents"]
                        if obj["Key"] != prefix and not obj["Key"].endswith("/")
                    ]
                    logger.info(f"Found {len(files)} files at {prefix}")
                    for file in files[:max_objects]:
                        file_name = (
                            file["Key"][len(prefix) :] if prefix else file["Key"]
                        )
                        logger.info(f"File: {file_name} ({file['Size']} bytes)")

                        # Check for HuggingFace model files
                        if file_name in [
                            "config.json",
                            "pytorch_model.bin",
                            "model.safetensors",
                            "tokenizer.json",
                            "tokenizer_config.json",
                        ]:
                            logger.info(f"Found potential model file: {file['Key']}")
                else:
                    logger.info(f"No files found at {prefix}")

                # Special handling for potential model directories
                check_for_model(bucket_name, prefix, s3_client)

            except Exception as e:
                logger.error(f"Error listing objects at {prefix}: {e}")

        # Function to check if a directory contains a HuggingFace model
        def check_for_model(bucket_name, prefix, s3_client):
            model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
            found_files = []

            # Check for each potential model file
            for file in model_files:
                file_key = f"{prefix}{file}"
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=file_key)
                    found_files.append(file)
                except:
                    pass

            if len(found_files) > 0:
                logger.info(f"POTENTIAL MODEL FOUND at s3://{bucket_name}/{prefix}")
                logger.info(f"Found model files: {', '.join(found_files)}")
                logger.info(f"Use this path for loading: s3://{bucket_name}/{prefix}")

        # Start exploration
        if not prefix:
            # Explore the root of the bucket
            list_objects("")
        else:
            # Check if the prefix exists as a "directory"
            try:
                # Check if prefix exists as a direct path
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=prefix)
                    logger.info(f"Found object at exact path: {prefix}")
                    # If this succeeds, the prefix is a file, not a directory
                except:
                    # If the head_object fails, it might be a directory or might not exist
                    pass

                # List the objects to see if the prefix exists as a directory
                list_objects(prefix)

            except Exception as e:
                logger.error(f"Error accessing path {prefix}: {e}")

                # Try listing the parent directory
                parent_prefix = "/".join(prefix.split("/")[:-1])
                logger.info(f"Trying parent directory: {parent_prefix}")
                list_objects(parent_prefix)

    except Exception as e:
        logger.error(f"Failed to explore S3 path: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore S3 buckets to find HuggingFace models"
    )
    parser.add_argument(
        "--s3_path",
        type=str,
        required=True,
        help="S3 path to explore (s3://bucket/path)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively explore subdirectories"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=2,
        help="Maximum depth for recursive exploration",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=20,
        help="Maximum number of objects to list per directory",
    )
    parser.add_argument(
        "--aws_access_key", type=str, default=None, help="AWS access key for S3 access"
    )
    parser.add_argument(
        "--aws_secret_key", type=str, default=None, help="AWS secret key for S3 access"
    )

    args = parser.parse_args()

    # Set AWS credentials if provided
    if args.aws_access_key and args.aws_secret_key:
        os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_key
        logger.info("Using provided AWS credentials")

    # Explore the S3 path
    explore_s3_path(args.s3_path, args.recursive, args.max_depth, args.max_objects)


if __name__ == "__main__":
    main()
