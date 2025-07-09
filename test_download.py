#!/usr/bin/env python3
"""
Simple script to download and read a Parquet file from GCS
to verify the parser upload is working correctly.
"""

import os
import pandas as pd
from google.cloud import storage

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google-creds.json"

# Configuration
BUCKET_NAME = 'ethereum-execution-times'
BLOCK_NUMBER = 22323394  # Example block number - change this to a block you've uploaded

def download_and_read_parquet(block_number):
    """Download and display contents of a Parquet file from GCS."""
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Construct blob name
    blob_name = f"{block_number}.parquet"
    blob = bucket.blob(blob_name)
    
    # Check if blob exists
    if not blob.exists():
        print(f"Blob {blob_name} does not exist in bucket {BUCKET_NAME}")
        print("\nListing available files:")
        for blob in bucket.list_blobs(max_results=10):
            print(f"  - {blob.name}")
        return
    
    # Download to temporary file
    temp_file = f"/tmp/{blob_name}"
    blob.download_to_filename(temp_file)
    print(f"Downloaded {blob_name} to {temp_file}")
    
    # Read Parquet file
    df = pd.read_parquet(temp_file)
    
    # Display info
    print(f"\nFile contains {len(df)} transactions")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Show some statistics
    if len(df) > 0:
        print(f"\nStatistics:")
        print(f"Average IO time: {df['IO_time'].mean():.2f} ms")
        print(f"Average EVM time: {df['EVM_time'].mean():.2f} ms")
        print(f"Total transactions: {len(df)}")
    
    # Clean up
    os.remove(temp_file)
    print(f"\nCleaned up temporary file")

if __name__ == "__main__":
    # You can change this to any block number you've uploaded
    download_and_read_parquet(BLOCK_NUMBER)