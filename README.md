
# CUDA Learning Project

This repository contains scripts and code for learning CUDA programming using AWS EC2 GPU instances. The project provides automation scripts for setting up and configuring GPU-enabled EC2 instances, along with sample CUDA C++ code for experimentation.

## Overview

The main goal of this repository is to facilitate learning CUDA programming by:
- Automating the setup of GPU-enabled AWS EC2 instances
- Providing infrastructure-as-code for AWS resources
- Including sample CUDA C++ code for learning and experimentation

## AWS Infrastructure CLI

The project includes a Go-based CLI tool that automates the creation and management of AWS infrastructure for CUDA-Learn. This tool replaces the previous bash scripts with a more robust and feature-rich implementation.

### Features

- Creates a VPC with all necessary networking components (subnets, internet gateway, routing tables)
- Creates security groups for SSH access
- Generates SSH key pairs and stores them both locally and in AWS SecretsManager
- Launches EC2 instances with GPU support using the latest PyTorch AMI
- Provides a simple command-line interface with configurable options

### Prerequisites

- Go 1.16 or later
- AWS credentials configured (via environment variables, AWS CLI, or IAM role)
- AWS permissions for:
  - EC2 (VPC, subnets, security groups, instances)
  - SecretsManager

### Usage

The CLI tool can be built and run using the provided Makefile targets:

```bash
# Build the CLI tool
make cli

# Set up VPC infrastructure only
make vpc

# Set up EC2 instance only (requires VPC to exist)
make instance

# Set up both VPC and EC2 instance
./build/cuda-learn setup
```

#### Advanced Usage

You can run the CLI tool directly with various flags:

```bash
./build/cuda-learn setup --region us-east-1 --project my-project --vpc-cidr 192.168.0.0/16 --subnet-cidr 192.168.1.0/24 --key-name my-key --instance-type p3.2xlarge
```

#### Available Commands

- `setup`: Sets up both VPC and EC2 infrastructure
- `vpc`: Sets up only VPC infrastructure
- `instance`: Sets up only EC2 instance (requires VPC to exist)

#### Available Flags

- `--region`: AWS region (default: us-west-2)
- `--project`: Project tag value (default: cuda-learn)
- `--vpc-cidr`: VPC CIDR block (default: 10.0.0.0/16)
- `--subnet-cidr`: Subnet CIDR block (default: 10.0.1.0/24)
- `--key-name`: SSH key name (default: gpu-key)
- `--instance-type`: EC2 instance type (default: g4dn.xlarge)

## Connecting to the EC2 Instance

### Direct SSH Connection
After running the CLI tool, you can connect using the provided command:
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

- `go-aws-cli/`: Go CLI tool for AWS infrastructure management
  - `cmd/`: Contains the main application entry point
  - `pkg/`: Contains packages for VPC, EC2, and common utilities
- `build/`: Contains compiled binaries
  - `cuda-learn`: The CLI tool binary
  - `matrix_gen`: CUDA sample application
- `Makefile`: Build configuration for both CUDA code and CLI tool
- `src/`: Sample CUDA C++ code (demonstration purposes)

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
2. Build and run the CLI tool:
   ```bash
   # Build the CLI tool
   make cli
   
   # Set up both VPC and EC2 infrastructure
   make vpc
   make instance
   
   # Or set up everything at once
   ./build/cuda-learn setup
   ```
3. SSH into the instance using the provided connection details
4. Clone this repository and build the CUDA code using make

## License

[Add your license information here]
