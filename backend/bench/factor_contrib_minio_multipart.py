"""Supplementary: compare boto3 single-stream get_object vs TransferManager
(multipart parallel download) against MinIO. The headline question is
whether the slow path on real AWS S3 (single-stream ~90 MB/s) shows up
on MinIO localhost too, or whether loopback bandwidth masks it.

Quick — float64 + parquet only (the user's stated config). 3 iters each.
"""
from __future__ import annotations
import io
import os
import sys
import time
import boto3
import botocore
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from boto3.s3.transfer import TransferConfig

ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
AK = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
SK = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.environ.get("MINIO_BUCKET", "factor-contrib")

cfg = botocore.config.Config(
    signature_version="s3v4",
    s3={"addressing_style": "path"},
    retries={"max_attempts": 1, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60,
    max_pool_connections=20,
)
s3 = boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=AK, aws_secret_access_key=SK, config=cfg)

key_F = "F_float64.parquet"
key_X = "X_float64.parquet"

# Confirm artefacts exist (uploaded by the main bench)
for k in [key_F, key_X]:
    s3.head_object(Bucket=BUCKET, Key=k)


def fetch_simple(key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def fetch_multipart(key: str) -> bytes:
    """Use TransferConfig with small multipart_chunksize to force the
    SDK to issue multiple ranged GETs in parallel."""
    cfg2 = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,    # default
        multipart_chunksize=8 * 1024 * 1024,    # 8 MB parts -> ~10 parts for an 80 MB file
        max_concurrency=10,
        use_threads=True,
    )
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buf, Config=cfg2)
    return buf.getvalue()


def time_fetch(label: str, fn, n: int = 3):
    for it in range(n):
        t0 = time.perf_counter()
        b = fn()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {label:<20} iter {it}  {ms:7.1f} ms   ({len(b)/1e6:.1f} MB)")


print("Single-stream get_object  (default boto3):")
print(f"  -- F ({s3.head_object(Bucket=BUCKET, Key=key_F)['ContentLength']/1e6:.1f} MB) --")
time_fetch("single-stream F", lambda: fetch_simple(key_F))
print(f"  -- X --")
time_fetch("single-stream X", lambda: fetch_simple(key_X))

print()
print("Multipart download_fileobj  (TransferConfig 8 MB parts, 10 concurrent):")
time_fetch("multipart F", lambda: fetch_multipart(key_F))
time_fetch("multipart X", lambda: fetch_multipart(key_X))
