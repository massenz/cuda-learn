/*
 * Copyright (c) 2025 AlertAvert.com.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Author: Marco Massenzio (marco@alertavert.com)
 */

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/cuda-learn/go-aws-cli/pkg/common"
	"github.com/cuda-learn/go-aws-cli/pkg/ec2"
	"github.com/cuda-learn/go-aws-cli/pkg/vpc"
	"github.com/spf13/cobra"
)

func main() {
	// Create root command
	var rootCmd = &cobra.Command{
		Use:   "cuda-learn",
		Short: "AWS CLI tool for CUDA-Learn project",
		Long:  `A CLI tool to create and manage AWS infrastructure for CUDA-Learn project.`,
	}

	// Define flags
	var region string
	var projectTag string
	var vpcCidr string
	var subnetCidr string
	var keyName string
	var instanceType string
	var ghAuthKeyPath string
	var ghAuthKeyName string

	// Set default values
	rootCmd.PersistentFlags().StringVar(&region, "region", "us-west-2", "AWS region")
	rootCmd.PersistentFlags().StringVar(&projectTag, "project", "cuda-learn", "Project tag value")
	rootCmd.PersistentFlags().StringVar(&vpcCidr, "vpc-cidr", "10.0.0.0/16", "VPC CIDR block")
	rootCmd.PersistentFlags().StringVar(&subnetCidr, "subnet-cidr", "10.0.1.0/24", "Subnet CIDR block")
	rootCmd.PersistentFlags().StringVar(&keyName, "key-name", "gpu-key", "SSH key name")
	rootCmd.PersistentFlags().StringVar(&instanceType, "instance-type", "g4dn.xlarge", "EC2 instance type")
	rootCmd.PersistentFlags().StringVar(&ghAuthKeyPath, "gh-auth", "", "Path to PEM key for GitHub authentication")
	rootCmd.PersistentFlags().StringVar(&ghAuthKeyName, "gh-auth-key-name", "gh-auth", "Name for GitHub authentication key in SecretsManager")

	// Create setup command (sets up both VPC and EC2)
	var setupCmd = &cobra.Command{
		Use:   "setup",
		Short: "Setup AWS infrastructure",
		Long:  `Setup AWS infrastructure including VPC and EC2 instance.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Initialize AWS config
			cfg, err := common.InitAWSConfig(region)
			if err != nil {
				return fmt.Errorf("failed to initialize AWS config: %v", err)
			}

			// Create VPC infrastructure
			vpcClient := vpc.NewVPCClient(cfg)
			vpcID, subnetID, instanceProfileArn, err := vpcClient.SetupVPC(projectTag, vpcCidr, subnetCidr)
			if err != nil {
				return fmt.Errorf("failed to setup VPC: %v", err)
			}

			// Create EC2 instance
			ec2Client := ec2.NewEC2Client(cfg)
			instanceID, publicIP, err := ec2Client.SetupEC2(projectTag, vpcID, subnetID, keyName, instanceType, instanceProfileArn)
			if err != nil {
				return fmt.Errorf("failed to setup EC2 instance: %v", err)
			}

			// Handle GitHub authentication key if provided
			if ghAuthKeyPath != "" {
				// Read the PEM key file
				privateKeyPEM, err := os.ReadFile(ghAuthKeyPath)
				if err != nil {
					return fmt.Errorf("failed to read GitHub auth key file: %v", err)
				}

				// Store the key in SecretsManager
				err = ec2Client.StoreKeyPEMInSecretsManager(ghAuthKeyName, privateKeyPEM)
				if err != nil {
					return fmt.Errorf("failed to store GitHub auth key in SecretsManager: %v", err)
				}
				fmt.Printf("GitHub authentication key stored in SecretsManager as: %s\n", ghAuthKeyName)
			}

			fmt.Printf("Successfully created infrastructure:\n")
			fmt.Printf("VPC ID: %s\n", vpcID)
			fmt.Printf("Subnet ID: %s\n", subnetID)
			fmt.Printf("IAM Instance Profile: %s\n", instanceProfileArn)
			fmt.Printf("Instance ID: %s\n", instanceID)
			fmt.Printf("Public IP: %s\n", publicIP)
			fmt.Printf("To SSH into the instance use:\n")
			fmt.Printf("  ssh -i private/%s.pem ubuntu@%s\n", keyName, publicIP)
			fmt.Printf("\nTo copy and run the setup script:\n")
			fmt.Printf("  scp -i private/%s.pem scripts/setup-host.sh ubuntu@%s:~/\n", keyName, publicIP)
			fmt.Printf("  ssh -i private/%s.pem ubuntu@%s \"chmod +x ~/setup-host.sh && ~/setup-host.sh\"\n", keyName, publicIP)

			return nil
		},
	}

	// Create vpc command (sets up only VPC)
	var vpcCmd = &cobra.Command{
		Use:   "vpc",
		Short: "Setup VPC infrastructure",
		Long:  `Setup VPC infrastructure including subnet, internet gateway, and routing.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Initialize AWS config
			cfg, err := common.InitAWSConfig(region)
			if err != nil {
				return fmt.Errorf("failed to initialize AWS config: %v", err)
			}

			// Create VPC infrastructure
			vpcClient := vpc.NewVPCClient(cfg)
			vpcID, subnetID, instanceProfileArn, err := vpcClient.SetupVPC(projectTag, vpcCidr, subnetCidr)
			if err != nil {
				return fmt.Errorf("failed to setup VPC: %v", err)
			}

			fmt.Printf("Successfully created VPC infrastructure:\n")
			fmt.Printf("VPC ID: %s\n", vpcID)
			fmt.Printf("Subnet ID: %s\n", subnetID)
			fmt.Printf("IAM Instance Profile ARN: %s\n", instanceProfileArn)

			return nil
		},
	}

	// Create instance command (sets up only EC2 instance)
	var instanceCmd = &cobra.Command{
		Use:   "instance",
		Short: "Setup EC2 instance",
		Long:  `Setup EC2 instance in the existing VPC.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Initialize AWS config
			cfg, err := common.InitAWSConfig(region)
			if err != nil {
				return fmt.Errorf("failed to initialize AWS config: %v", err)
			}

			// Find VPC by tag
			vpcClient := vpc.NewVPCClient(cfg)
			vpcID, subnetID, instanceProfileArn, err := vpcClient.SetupVPC(projectTag, vpcCidr, subnetCidr)
			if err != nil {
				return fmt.Errorf("failed to find VPC: %v", err)
			}

			if vpcID == "" {
				return fmt.Errorf("no VPC found with tag project=%s, please create VPC first", projectTag)
			}

			// Create EC2 instance
			ec2Client := ec2.NewEC2Client(cfg)
			instanceID, publicIP, err := ec2Client.SetupEC2(projectTag, vpcID, subnetID, keyName, instanceType, instanceProfileArn)
			if err != nil {
				return fmt.Errorf("failed to setup EC2 instance: %v", err)
			}

			// Handle GitHub authentication key if provided
			if ghAuthKeyPath != "" {
				// Read the PEM key file
				privateKeyPEM, err := os.ReadFile(ghAuthKeyPath)
				if err != nil {
					return fmt.Errorf("failed to read GitHub auth key file: %v", err)
				}

				// Store the key in SecretsManager
				err = ec2Client.StoreKeyPEMInSecretsManager(ghAuthKeyName, privateKeyPEM)
				if err != nil {
					return fmt.Errorf("failed to store GitHub auth key in SecretsManager: %v", err)
				}
				fmt.Printf("GitHub authentication key stored in SecretsManager as: %s\n", ghAuthKeyName)
			}

			fmt.Printf("Successfully created EC2 instance:\n")
			fmt.Printf("IAM Instance Profile: %s\n", instanceProfileArn)
			fmt.Printf("Instance ID: %s\n", instanceID)
			fmt.Printf("Public IP: %s\n", publicIP)
			fmt.Printf("To SSH into the instance use:\n")
			fmt.Printf("  ssh -i private/%s.pem ubuntu@%s\n", keyName, publicIP)
			fmt.Printf("\nTo copy and run the setup script:\n")
			fmt.Printf("  scp -i private/%s.pem scripts/setup-host.sh ubuntu@%s:~/\n", keyName, publicIP)
			fmt.Printf("  ssh -i private/%s.pem ubuntu@%s \"chmod +x ~/setup-host.sh && ~/setup-host.sh\"\n", keyName, publicIP)

			return nil
		},
	}

	// Create teardown command
	var teardownCmd = &cobra.Command{
		Use:   "teardown",
		Short: "Terminate EC2 instance",
		Long:  `Terminate EC2 instance with the specified project tag.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Initialize AWS config
			cfg, err := common.InitAWSConfig(region)
			if err != nil {
				return fmt.Errorf("failed to initialize AWS config: %v", err)
			}

			// Create EC2 client
			ec2Client := ec2.NewEC2Client(cfg)

			// Find instances by project tag
			instances, err := ec2Client.FindInstancesByProjectTag(projectTag)
			if err != nil {
				return fmt.Errorf("failed to find instances: %v", err)
			}

			// Handle the three possible outcomes
			switch len(instances) {
			case 0:
				// No instances found
				common.LogError("No instances found with project tag: %s", projectTag)
				return fmt.Errorf("no instances found with project tag: %s", projectTag)

			case 1:
				// One instance found, ask for confirmation
				instance := instances[0]
				fmt.Println("Found one instance with the specified project tag:")
				fmt.Println(ec2Client.GetInstanceDetails(instance))

				// Ask for confirmation
				fmt.Print("Do you want to terminate this instance? (yes/no): ")
				var response string
				fmt.Scanln(&response)

				if strings.ToLower(response) == "yes" {
					// Terminate the instance
					err := ec2Client.TerminateInstance(*instance.InstanceId)
					if err != nil {
						return fmt.Errorf("failed to terminate instance: %v", err)
					}
					fmt.Println("Instance termination initiated successfully.")
				} else {
					fmt.Println("Instance termination cancelled.")
				}

			default:
				// Multiple instances found
				fmt.Printf("Found %d instances with the specified project tag:\n\n", len(instances))
				for _, instance := range instances {
					fmt.Println(ec2Client.GetInstanceDetails(instance))
					fmt.Println("---")
				}
				fmt.Println("Multiple instances found. Please specify a unique instance to terminate.")
			}

			return nil
		},
	}

	// Add commands to root
	rootCmd.AddCommand(setupCmd)
	rootCmd.AddCommand(vpcCmd)
	rootCmd.AddCommand(instanceCmd)
	rootCmd.AddCommand(teardownCmd)

	// Execute
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
