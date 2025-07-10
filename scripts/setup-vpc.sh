#!/usr/bin/env zsh
set -eu

source ${COMMON_UTILS}/utils.sh

# Check if a VPC with Tag Key=project, Value=cuda-learn exists
VPC_ID=$(aws ec2 describe-vpcs \
  --region us-west-2 \
  --filters "Name=tag:project,Values=cuda-learn" \
  --query "Vpcs[0].VpcId" \
  --output text)

echo "VPC_ID=$VPC_ID"

# If exists, use it
if [ "$VPC_ID" != "None" ]; then
  echo "Found existing VPC: $VPC_ID"
else
  echo "No existing VPC found. Creating new one..."

  # Create new VPC in us-west-2
  VPC_ID=$(aws ec2 create-vpc \
    --region us-west-2 \
    --cidr-block 10.0.0.0/16 \
    --query "Vpc.VpcId" \
    --output text)

  # Add tag
  aws ec2 create-tags \
    --region us-west-2 \
    --resources $VPC_ID \
    --tags Key=project,Value=cuda-learn

  echo "Created VPC: $VPC_ID"

  # Create subnet in us-west-2a
  SUBNET_ID=$(aws ec2 create-subnet \
    --region us-west-2 \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-west-2a \
    --query "Subnet.SubnetId" \
    --output text)

  aws ec2 create-tags \
  --region us-west-2 \
  --resources $SUBNET_ID \
  --tags Key=project,Value=cuda-learn

  echo "Created Subnet: $SUBNET_ID"

  # Create internet gateway
  IGW_ID=$(aws ec2 create-internet-gateway \
    --region us-west-2 \
    --query "InternetGateway.InternetGatewayId" \
    --output text)

  aws ec2 attach-internet-gateway \
    --region us-west-2 \
    --internet-gateway-id $IGW_ID \
    --vpc-id $VPC_ID

  echo "Created and attached Internet Gateway: $IGW_ID"

  # Create route table
  RTB_ID=$(aws ec2 create-route-table \
    --region us-west-2 \
    --vpc-id $VPC_ID \
    --query "RouteTable.RouteTableId" \
    --output text)

  aws ec2 create-route \
    --region us-west-2 \
    --route-table-id $RTB_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID

  aws ec2 associate-route-table \
    --region us-west-2 \
    --route-table-id $RTB_ID \
    --subnet-id $SUBNET_ID

  echo "Created Route Table: $RTB_ID and associated with Subnet: $SUBNET_ID"

  # Optionally: enable auto-assign public IP
  aws ec2 modify-subnet-attribute \
    --region us-west-2 \
    --subnet-id $SUBNET_ID \
    --map-public-ip-on-launch

  echo "Enabled auto-assign public IP for Subnet: $SUBNET_ID"
fi
