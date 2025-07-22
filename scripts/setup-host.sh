#!/usr/bin/env bash
#
# Copyright (c) 2025 AlertAvert.com.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)
set -eu

git config --global init.defaultBranch main
git config --global user.name "Marco Massenzio"
git config --global user.email "marco@massenz.io"

# Download the private key from Secrets Manager
SECRET_NAME=${1:-"gh-auth"}
OUTPUT_FILE=~/.ssh/${SECRET_NAME}.pem
AWS_REGION=$(curl -s --connect-timeout 2 \
    http://169.254.169.254/latest/meta-data/placement/region || echo "us-west-2")

echo "Downloading secret key '${SECRET_NAME}' from AWS Secrets Manager."
echo "Using AWS region: ${AWS_REGION}"

mkdir -p ~/.ssh
aws secretsmanager get-secret-value --secret-id "${SECRET_NAME}" \
    --region "${AWS_REGION}" \
    --query SecretString --output text > "${OUTPUT_FILE}"
chmod 600 "${OUTPUT_FILE}"

eval "$(ssh-agent -s)"
ssh-add "${OUTPUT_FILE}"

git clone git@github.com:massenz/cuda-learn.git && \
    cd cuda-learn && \
    make all
