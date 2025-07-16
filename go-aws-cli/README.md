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
   git clone https://github.com/cuda-learn/go-aws-cli.git
   cd go-aws-cli
   ```

2. Build the CLI:
   ```
   go build -o aws-cli ./cmd
   ```

## Usage

### Basic Usage

To create the infrastructure with default settings:

```
./aws-cli setup
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
./aws-cli setup --region us-east-1 --project my-project --vpc-cidr 192.168.0.0/16 --subnet-cidr 192.168.1.0/24 --key-name my-key --instance-type p3.2xlarge
```

### Available Flags

- `--region`: AWS region (default: us-west-2)
- `--project`: Project tag value (default: cuda-learn)
- `--vpc-cidr`: VPC CIDR block (default: 10.0.0.0/16)
- `--subnet-cidr`: Subnet CIDR block (default: 10.0.1.0/24)
- `--key-name`: SSH key name (default: gpu-key)
- `--instance-type`: EC2 instance type (default: g4dn.xlarge)

## SSH Access

After creating the infrastructure, the CLI will output the SSH command to connect to the instance:

```
To SSH into the instance use:
  ssh -i private/gpu-key.pem ubuntu@<public-ip>
```

## Development

### Project Structure

- `cmd/`: Contains the main application entry point
- `pkg/common/`: Common utilities and AWS configuration
- `pkg/vpc/`: VPC-related functionality
- `pkg/ec2/`: EC2-related functionality

### Building from Source

```
go build -o aws-cli ./cmd
```

### Running Tests

```
go test ./...
```

## License

MIT
