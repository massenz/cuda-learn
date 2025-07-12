
# CUDA Learning Project

This repository contains scripts and code for learning CUDA programming using AWS EC2 GPU instances. The project provides automation scripts for setting up and configuring GPU-enabled EC2 instances, along with sample CUDA C++ code for experimentation.

## Overview

The main goal of this repository is to facilitate learning CUDA programming by:
- Automating the setup of GPU-enabled AWS EC2 instances
- Providing infrastructure-as-code for AWS resources
- Including sample CUDA C++ code for learning and experimentation

## AWS Setup Scripts

### setup-vpc.sh
This script handles the one-time setup of the AWS VPC infrastructure:
- Creates a new VPC with CIDR block 10.0.0.0/16 in us-west-2 region
- Sets up a subnet in us-west-2a (CIDR: 10.0.1.0/24)
- Creates and attaches an Internet Gateway
- Configures route tables for internet access
- Enables auto-assign public IP for the subnet
- Tags all resources with project=cuda-learn

Note: This script only needs to be run once to set up the initial infrastructure.

### setup-ec2.sh
This script automates the creation and configuration of a GPU-enabled EC2 instance:
- Creates or reuses an SSH key pair (`gpu-key`) for instance access
- Validates VPC and subnet configuration using the cuda-learn project tags
- Sets up a security group (`ssh-sg`) that allows inbound SSH access (port 22)
- Launches a GPU-enabled instance (g4dn.xlarge) using the latest PyTorch AMI
- Waits for the instance to be running and retrieves its public IP
- Provides SSH connection details upon completion

## Connecting to the EC2 Instance

### Direct SSH Connection
After running setup-ec2.sh, you can connect using the provided command:
```bash
ssh -i private/gpu-key.pem ubuntu@<PUBLIC_IP>
```

### SSH Config Setup
For convenience, you can configure your `~/.ssh/config` file to create an alias:

```
Host cuda-learn
    HostName cuda-learn
    User ubuntu
    IdentityFile ~/.ssh/gpu-key.pem
    StrictHostKeyChecking no
```
after adding this line to your `/etc/hosts`file:
```
<PUBLIC_IP> cuda-learn
```

After adding this configuration, you can simply connect using:
```bash
ssh cuda-learn
```

Remember to update the `HostName`(or `/etc/hosts`) whenever you create a new instance. You can find the current public IP in the setup-ec2.sh output or via the AWS Console.

## AWS GPU Instance Quota Requirements

Before running the scripts, ensure you have the appropriate AWS service quotas for GPU instances:

1. Visit the [AWS Service Quotas Console](https://console.aws.amazon.com/servicequotas/)
2. Navigate to EC2 service quotas
3. Search for "Running On-Demand G instances"
4. Request a quota increase if the current limit is 0
5. Additionally, search for "All G Spot Instance Requests" and request an increase

Important Links:
- [AWS Service Quotas Documentation](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html)
- [EC2 G-type Instance Information](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [GPU Instance Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)

## Project Structure

- `setup-ec2.sh`: Main script for EC2 instance provisioning
- `Makefile`: Build configuration for CUDA C++ code
- Sample CUDA C++ code (demonstration purposes)

## Build System

The project includes a Makefile for building CUDA C++ code. The build system handles compilation and linking of CUDA source files.

## TODO

Future Improvements:
- [ ] Move GitHub private key storage from local file to AWS Secrets Manager
  - This will improve security and enable direct access from EC2 instances
  - Currently, the key is stored locally and uploaded via scp in setup-ec2.sh

## Prerequisites

- AWS CLI configured with appropriate credentials
- Sufficient AWS quotas for GPU instances (both Spot and On-Demand)
- CUDA toolkit (for local development)
- Make build system

## Usage

1. Request AWS GPU instance quota increases if needed
2. Run the setup script:
   ```bash
   ./setup-ec2.sh
   ```
3. SSH into the instance using the provided connection details
4. Clone this repository and build the CUDA code using make

## License

[Add your license information here]
