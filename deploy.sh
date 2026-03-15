#!/bin/bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-west-2}"
STACK_NAME="medcoverage"
ECR_REPO="medcoverage-api"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
DATA_BUCKET="medcoverage-data-${AWS_ACCOUNT_ID}"

echo "=== MedCoverage AWS Deploy (EC2) ==="
echo "Account: ${AWS_ACCOUNT_ID}"
echo "Region:  ${AWS_REGION}"
echo ""

# ── 1. ECR ─────────────────────────────────────────────
echo "[1/5] Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# ── 2. Build & Push ────────────────────────────────────
echo "[2/5] Building and pushing Docker image..."
docker build --platform linux/amd64 -t ${ECR_REPO}:latest -f Dockerfile.prod .
docker tag ${ECR_REPO}:latest ${IMAGE_URI}
docker push ${IMAGE_URI}

# ── 3. S3 Data ─────────────────────────────────────────
echo "[3/5] Uploading data to S3..."
aws s3 mb s3://${DATA_BUCKET} --region ${AWS_REGION} 2>/dev/null || true
aws s3 sync data/embeddings/ s3://${DATA_BUCKET}/embeddings/ --quiet
aws s3 sync data/processed/ s3://${DATA_BUCKET}/processed/ --quiet
aws s3 sync backend/data/sample-lancedb/ s3://${DATA_BUCKET}/sample-lancedb/ --quiet
aws s3 sync data/sample_data/ s3://${DATA_BUCKET}/sample_data/ --quiet

# ── 4. CloudFormation ──────────────────────────────────
echo "[4/5] Deploying CloudFormation stack (EC2 + ALB + Redis)..."
aws cloudformation deploy \
  --template-file infra/cloudformation.yml \
  --stack-name ${STACK_NAME} \
  --region ${AWS_REGION} \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ImageUri=${IMAGE_URI} \
    DataS3Bucket=${DATA_BUCKET}

# ── 5. Output ──────────────────────────────────────────
echo "[5/5] Done!"
ALB_DNS=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${AWS_REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' --output text)
EC2_IP=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${AWS_REGION} \
  --query 'Stacks[0].Outputs[?OutputKey==`EC2PublicIP`].OutputValue' --output text)
echo ""
echo "========================================="
echo "App URL:  http://${ALB_DNS}"
echo "Health:   http://${ALB_DNS}/health"
echo "EC2 SSH:  ssh ec2-user@${EC2_IP}"
echo "========================================="