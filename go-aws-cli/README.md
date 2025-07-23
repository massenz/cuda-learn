# AWS Infrastructure CLI for CUDA-Learn

This CLI tool automates the creation of AWS infrastructure for CUDA-Learn project, including VPC and EC2 instances with GPU support.

## Features

- Creates a VPC with all necessary networking components (subnets, internet gateway, routing tables)
- Creates security groups for SSH access
- Generates SSH key pairs and stores them both locally and in AWS SecretsManager
- Launches EC2 instances with GPU support using the latest PyTorch AMI
- Provides a simple command-line interface with configurable options

## Prerequisites

- Go 1.16 or later
- AWS credentials configured (via environment variables, AWS CLI, or IAM role)
- AWS permissions for:
  - EC2 (VPC, subnets, security groups, instances)
  - SecretsManager

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:massenz/cuda-learn.git
   cd cuda-learn
   ```

2. Build the CLI:
   ```
   make cli
   ```

## Usage

### Basic Usage

To create the infrastructure with default settings:

```
./build/cuda-learn setup
```

This will:
1. Create a VPC with CIDR 10.0.0.0/16 in us-west-2 region (or use existing one with tag project=cuda-learn)
2. Create a subnet with CIDR 10.0.1.0/24
3. Set up internet gateway and routing
4. Create a security group for SSH access
5. Generate an SSH key pair and store it in private/gpu-key.pem and AWS SecretsManager
6. Launch a g4dn.xlarge instance with the latest GPU-enabled PyTorch AMI

### Advanced Usage

You can customize the infrastructure creation with various flags:

```
./build/cuda-learn setup --region us-east-1 --project my-project --vpc-cidr 192.168.0.0/16 --subnet-cidr 192.168.1.0/24 --key-name my-key --instance-type p3.2xlarge
```

### Available Commands

- `setup`: Sets up both VPC and EC2 infrastructure
- `vpc`: Sets up only VPC infrastructure
- `instance`: Sets up only EC2 instance (requires VPC to exist)
- `teardown`: Terminates EC2 instances with the specified project tag

### Available Flags

- `--region`: AWS region (default: us-west-2)
- `--project`: Project tag value (default: cuda-learn)
- `--vpc-cidr`: VPC CIDR block (default: 10.0.0.0/16)
- `--subnet-cidr`: Subnet CIDR block (default: 10.0.1.0/24)
- `--key-name`: SSH key name for EC2 instance access (default: gpu-key)
- `--instance-type`: EC2 instance type (default: g4dn.xlarge)
- `--gh-auth`: Path to PEM key for GitHub authentication (optional)
- `--gh-auth-key-name`: Name for GitHub authentication key in SecretsManager (default: gh-auth)

### Teardown Command

The `teardown` command allows you to terminate EC2 instances with the specified project tag. It has additional flags for more control:

- `--instance`: Specifies an instance ID to terminate (must match the project tag)
- `--all`: Terminates all instances with matching project tag

If neither flag is specified and multiple instances match the project tag, an interactive selection menu will be displayed:

```
More than one instance matches the `project=cuda-learn` tag:
<< details of instances>>
Which one would you like to terminate:
1) id-12345
2) id-9876
3) id-88754

Please choose one (or simply Enter to exit): 
```

## SSH Access

After creating the infrastructure, the CLI will output the SSH command to connect to the instance:

```
To SSH into the instance use:
  ssh -i private/gpu-key.pem ubuntu@<public-ip>
```

## IAM Role for EC2 Instances

The CLI automatically creates and attaches an IAM role and instance profile to EC2 instances, enabling them to access AWS services without manual credential management.

### What's Created

- **IAM Role**: A role named `<project-tag>-ec2-role` with a trust policy allowing EC2 instances to assume it
- **Instance Profile**: A profile named `<project-tag>-profile` that's attached to the EC2 instance

### Permissions

The IAM role has the following managed policies attached:

- **AmazonSSMManagedInstanceCore**: Allows the instance to use AWS Systems Manager
- **SecretsManagerReadWrite**: Allows the instance to read and write secrets in AWS Secrets Manager

### Benefits

- **No Manual Credentials**: EC2 instances can access AWS services without storing credentials on the instance
- **Automatic Rotation**: AWS automatically rotates the temporary credentials
- **Secure**: Follows AWS best practices for security
- **Simplified Setup**: The `aws` CLI on the instance works without additional configuration

### Usage on the Instance

Once connected to the instance, you can use the AWS CLI without additional configuration:

```bash
# Example: Retrieve a secret from Secrets Manager
aws secretsmanager get-secret-value --secret-id "my-secret"

# Example: List S3 buckets
aws s3 ls
```

## Command Line Help

Below is the output of the `help` command showing all available commands and flags:

```
A CLI tool to create and manage AWS infrastructure for CUDA-Learn project.

Usage:
  cuda-learn [command]

Available Commands:
  help        Help about any command
  instance    Setup EC2 instance
  setup       Setup AWS infrastructure
  teardown    Terminate EC2 instance
  vpc         Setup VPC infrastructure

Flags:
      --gh-auth string              Path to PEM key for GitHub authentication
      --gh-auth-key-name string     Name for GitHub authentication key in SecretsManager (default "gh-auth")
  -h, --help                        help for cuda-learn
      --instance-type string        EC2 instance type (default "g4dn.xlarge")
      --key-name string             SSH key name (default "gpu-key")
      --project string              Project tag value (default "cuda-learn")
      --region string               AWS region (default "us-west-2")
      --subnet-cidr string          Subnet CIDR block (default "10.0.1.0/24")
      --vpc-cidr string             VPC CIDR block (default "10.0.0.0/16")

Use "cuda-learn [command] --help" for more information about a command.
```

For the `teardown` command specifically:

```
Terminate EC2 instance with the specified project tag.

Usage:
  cuda-learn teardown [flags]

Flags:
      --all                Terminate all instances with matching project tag
  -h, --help               help for teardown
      --instance string    Instance ID to terminate (must match project tag)

Global Flags:
      --gh-auth string              Path to PEM key for GitHub authentication
      --gh-auth-key-name string     Name for GitHub authentication key in SecretsManager (default "gh-auth")
      --instance-type string        EC2 instance type (default "g4dn.xlarge")
      --key-name string             SSH key name (default "gpu-key")
      --project string              Project tag value (default "cuda-learn")
      --region string               AWS region (default "us-west-2")
      --subnet-cidr string          Subnet CIDR block (default "10.0.1.0/24")
      --vpc-cidr string             VPC CIDR block (default "10.0.0.0/16")
```

## Development

### Project Structure

- `cmd/`: Contains the main application entry point
- `pkg/common/`: Common utilities and AWS configuration
- `pkg/vpc/`: VPC-related functionality
- `pkg/ec2/`: EC2-related functionality
- `pkg/iam/`: IAM role and instance profile functionality
