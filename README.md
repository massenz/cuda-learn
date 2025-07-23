
# CUDA Learning Project

This repository contains scripts and code for learning CUDA programming using AWS EC2 GPU instances. The project provides automation scripts for setting up and configuring GPU-enabled EC2 instances, along with sample CUDA C++ code for experimentation.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

The CLI is more fully described [here](go-aws-cli/README.md), and can be built with:
```shell
make cli
```
The full set of available commands and flags can be seen using:
```shell
./build/cuda-learn help
```

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

Remember to update the `HostName`(or `/etc/hosts`) whenever you create a new instance. You can find the current public IP via the AWS Console.

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

The two main commands are `build` and `run` (they can only be run on the GPU instance, as they require `nvcc`, the CUDA compiler) and require specifying the `SRC` file to compile:

```shell
# On the EC2 instance
$ make SRC=gpu-props.cu run
```

We recommend using an IDE such as VSCode connected remotely to the EC2 instance, please see [this blog](https://codetrips.com/2025/07/17/cuda-development-on-aws-gpu-instances/) for more details.

## Prerequisites

- AWS configured with appropriate credentials (at the very minimum, `AWS_PROFILE` pointing to a profile in `~/.aws/credentials` with sufficient permissions to create AMI roles, create a VPC, and run EC2 instances)
- Sufficient AWS quotas for GPU instances (both Spot and On-Demand)
- CUDA toolkit (this comes pre-installed in the GPU-enable AMI that we run on the EC2 instance)
- Make build system (for local development; on the remote instance it comes pre-installed)

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

This project is release under the Apache 2.0 License, see the LICENSE file in this directory for full details.
