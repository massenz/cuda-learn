package main

import (
	"fmt"
	"os"

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

	// Set default values
	rootCmd.PersistentFlags().StringVar(&region, "region", "us-west-2", "AWS region")
	rootCmd.PersistentFlags().StringVar(&projectTag, "project", "cuda-learn", "Project tag value")
	rootCmd.PersistentFlags().StringVar(&vpcCidr, "vpc-cidr", "10.0.0.0/16", "VPC CIDR block")
	rootCmd.PersistentFlags().StringVar(&subnetCidr, "subnet-cidr", "10.0.1.0/24", "Subnet CIDR block")
	rootCmd.PersistentFlags().StringVar(&keyName, "key-name", "gpu-key", "SSH key name")
	rootCmd.PersistentFlags().StringVar(&instanceType, "instance-type", "g4dn.xlarge", "EC2 instance type")

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
			vpcID, subnetID, err := vpcClient.SetupVPC(projectTag, vpcCidr, subnetCidr)
			if err != nil {
				return fmt.Errorf("failed to setup VPC: %v", err)
			}

			// Create EC2 instance
			ec2Client := ec2.NewEC2Client(cfg)
			instanceID, publicIP, err := ec2Client.SetupEC2(projectTag, vpcID, subnetID, keyName, instanceType)
			if err != nil {
				return fmt.Errorf("failed to setup EC2 instance: %v", err)
			}

			fmt.Printf("Successfully created infrastructure:\n")
			fmt.Printf("VPC ID: %s\n", vpcID)
			fmt.Printf("Subnet ID: %s\n", subnetID)
			fmt.Printf("Instance ID: %s\n", instanceID)
			fmt.Printf("Public IP: %s\n", publicIP)
			fmt.Printf("To SSH into the instance use:\n")
			fmt.Printf("  ssh -i private/%s.pem ubuntu@%s\n", keyName, publicIP)

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
			vpcID, subnetID, err := vpcClient.SetupVPC(projectTag, vpcCidr, subnetCidr)
			if err != nil {
				return fmt.Errorf("failed to setup VPC: %v", err)
			}

			fmt.Printf("Successfully created VPC infrastructure:\n")
			fmt.Printf("VPC ID: %s\n", vpcID)
			fmt.Printf("Subnet ID: %s\n", subnetID)

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
			vpcID, subnetID, err := vpcClient.SetupVPC(projectTag, vpcCidr, subnetCidr)
			if err != nil {
				return fmt.Errorf("failed to find VPC: %v", err)
			}

			if vpcID == "" {
				return fmt.Errorf("no VPC found with tag project=%s, please create VPC first", projectTag)
			}

			// Create EC2 instance
			ec2Client := ec2.NewEC2Client(cfg)
			instanceID, publicIP, err := ec2Client.SetupEC2(projectTag, vpcID, subnetID, keyName, instanceType)
			if err != nil {
				return fmt.Errorf("failed to setup EC2 instance: %v", err)
			}

			fmt.Printf("Successfully created EC2 instance:\n")
			fmt.Printf("Instance ID: %s\n", instanceID)
			fmt.Printf("Public IP: %s\n", publicIP)
			fmt.Printf("To SSH into the instance use:\n")
			fmt.Printf("  ssh -i private/%s.pem ubuntu@%s\n", keyName, publicIP)

			return nil
		},
	}

	// Add commands to root
	rootCmd.AddCommand(setupCmd)
	rootCmd.AddCommand(vpcCmd)
	rootCmd.AddCommand(instanceCmd)

	// Execute
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
