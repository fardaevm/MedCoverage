#!/bin/bash
set -e

# Sync data from S3 if bucket is configured
if [ -n "$DATA_S3_BUCKET" ]; then
  echo "[entrypoint] Syncing data from s3://${DATA_S3_BUCKET}..."
  pip install awscli --quiet 2>/dev/null || true
  aws s3 sync "s3://${DATA_S3_BUCKET}/embeddings/" /app/data/embeddings/ --quiet
  aws s3 sync "s3://${DATA_S3_BUCKET}/processed/" /app/data/processed/ --quiet
  aws s3 sync "s3://${DATA_S3_BUCKET}/sample-lancedb/" /app/backend/data/sample-lancedb/ --quiet
  aws s3 sync "s3://${DATA_S3_BUCKET}/sample_data/" /app/data/sample_data/ --quiet
  echo "[entrypoint] Data sync complete."
fi

exec "$@"