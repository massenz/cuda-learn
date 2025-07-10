#!/usr/bin/env zsh
set -eu

source ${COMMON_UTILS}/utils.sh

KEY_NAME=gpu-key
# ✅ Check if a key pair named `gpu-key` exists
KEY_ID=$(aws ec2 describe-key-pairs --key-names ${KEY_NAME} \
  --query 'KeyPairs[*].KeyPairId' --output text)

if [[ -z ${KEY_ID} ]]; then
  aws ec2 create-key-pair --key-name ${KEY_NAME} \
    --query 'KeyMaterial' --output text > ${KEY_NAME}.pem
  chmod 400 ${KEY_NAME}.pem
  KEY_ID=$(aws ec2 describe-key-pairs --key-names ${KEY_NAME} \
    --query 'KeyPairs[*].KeyPairId' --output text)

  success "Created SSH key  ${KEY_NAME} (${KEY_ID})"
fi
msg "Key pair in ${KEY_NAME}.pem"

VPC_ID=$(aws ec2 describe-vpcs \
  --region us-west-2 \
  --filters "Name=tag:project,Values=cuda-learn" \
  --query "Vpcs[0].VpcId" \
  --output text)

if [[ ${VPC_ID} == "None" ]]; then
  fatal "Cannot find VPC for Project cuda-learn"
fi
msg "VPC ID: ${VPC_ID}"

SUBNET_ID=$(aws ec2 describe-subnets \
  --region us-west-2 \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:project,Values=cuda-learn" \
  --query "Subnets[0].SubnetId" \
  --output text)
if [[ ${SUBNET_ID} == "None" ]]; then
  fatal "Cannot find a suitable Subnet in VPC for Project cuda-learn"
fi
msg "Subnet ID: ${SUBNET_ID}"

SG_ID=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=${VPC_ID}" \
  --query 'SecurityGroups[*].[GroupId,GroupName,Description]' --output text | \
  grep ssh-sg | cut -f 1)
if [[ -z ${SG_ID} ]]; then
  msg "Creating Security Group and enabling SSH access..."
  SG_ID=$(aws ec2 create-security-group --group-name ssh-sg \
    --description "SSH access" --vpc-id ${VPC_ID})
  aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} --protocol tcp \
    --port 22 --cidr 0.0.0.0/0
  success "Created SG for SSH Access: ${SG_ID}"
fi
msg "SSH Access enabled (Security Group: ${SG_ID})"

SUBNET_ID=$(aws ec2 describe-subnets --query 'Subnets[0].SubnetId' --output text)
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI GPU*" \
  --query 'Images[*].[ImageId,CreationDate]' \
  --output text | sort -k2 -r | head -n1 | cut -f 1)

# Cheapest GPU-enabled AMI
# See: https://aws.amazon.com/ec2/pricing/on-demand/
INSTANCE_TYPE=g4dn.xlarge
msg "Reserving an EC2 Instance (${INSTANCE_TYPE}), Deep Learning AMI GPU: ${AMI_ID}"

aws ec2 run-instances \
  --image-id ${AMI_ID} \
  --count 1 \
  --instance-type ${INSTANCE_TYPE} \
  --key-name ${KEY_NAME} \
  --security-group-ids ${SG_ID} \
  --subnet-id ${SUBNET_ID} \
  --output text
