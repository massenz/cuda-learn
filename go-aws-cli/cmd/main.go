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
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// Get verbose flag
			verbose, _ := cmd.Flags().GetBool("verbose")

			// Initialize logger
			if err := common.InitLogger(verbose); err != nil {
				return fmt.Errorf("failed to initialize logger: %w", err)
			}

			return nil
		},
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
	var verbose bool

	// Set default values
	rootCmd.PersistentFlags().StringVar(&region, "region", "us-west-2", "AWS region")
	rootCmd.PersistentFlags().StringVar(&projectTag, "project", "cuda-learn", "Project tag value")
	rootCmd.PersistentFlags().StringVar(&vpcCidr, "vpc-cidr", "10.0.0.0/16", "VPC CIDR block")
	rootCmd.PersistentFlags().StringVar(&subnetCidr, "subnet-cidr", "10.0.1.0/24", "Subnet CIDR block")
	rootCmd.PersistentFlags().StringVar(&keyName, "key-name", "gpu-key", "SSH key name")
	rootCmd.PersistentFlags().StringVar(&instanceType, "instance-type", "g4dn.xlarge", "EC2 instance type")
	rootCmd.PersistentFlags().StringVar(&ghAuthKeyPath, "gh-auth", "", "Path to PEM key for GitHub authentication")
	rootCmd.PersistentFlags().StringVar(&ghAuthKeyName, "gh-auth-key-name", "gh-auth", "Name for GitHub authentication key in SecretsManager")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose logging to console")

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

			// Check if no instances found
			if len(instances) == 0 {
				errMsg := fmt.Errorf("no instances found with project tag: %s", projectTag)
				common.LogError(errMsg, "")
				fmt.Printf("No instances found with project tag: %s\n", projectTag)
				return nil
			}

			// Get flag values
			specifiedInstanceID, _ := cmd.Flags().GetString("instance")
			terminateAll, _ := cmd.Flags().GetBool("all")

			// Check if both --all and --instance flags are specified
			if specifiedInstanceID != "" && terminateAll {
				err := fmt.Errorf("cannot specify both --all and --instance flags")
				common.LogError(err, "Invalid flag combination")
				return err
			}

			// If instance ID is specified, find and terminate that specific instance
			if specifiedInstanceID != "" {
				var found bool
				var targetInstance int

				for i, instance := range instances {
					if *instance.InstanceId == specifiedInstanceID {
						targetInstance = i
						found = true
						break
					}
				}

				if !found {
					errMsg := fmt.Errorf("no instance with ID %s found with project tag: %s", specifiedInstanceID, projectTag)
					common.LogError(errMsg, "Instance not found")
					fmt.Printf("No instance with ID %s found with project tag: %s\n", specifiedInstanceID, projectTag)
					return nil
				}

				fmt.Println("Found the specified instance:")
				fmt.Println(ec2Client.GetInstanceDetails(instances[targetInstance]))

				// Ask for confirmation
				fmt.Print("Do you want to terminate this instance? (yes/no): ")
				var response string
				fmt.Scanln(&response)

				if strings.ToLower(response) == "yes" {
					// Terminate the instance
					err := ec2Client.TerminateInstance(specifiedInstanceID)
					if err != nil {
						return fmt.Errorf("failed to terminate instance: %v", err)
					}
					fmt.Println("Instance termination initiated successfully.")
				} else {
					fmt.Println("Instance termination cancelled.")
				}

				return nil
			}

			// If --all flag is specified, terminate all instances
			if terminateAll {
				fmt.Printf("Found %d instances with project tag %s:\n\n", len(instances), projectTag)
				for _, instance := range instances {
					fmt.Println(ec2Client.GetInstanceDetails(instance))
					fmt.Println("---")
				}

				// Ask for confirmation
				fmt.Printf("Do you want to terminate all %d instances? (yes/no): ", len(instances))
				var response string
				fmt.Scanln(&response)

				if strings.ToLower(response) == "yes" {
					// Terminate all instances
					for _, instance := range instances {
						instanceID := *instance.InstanceId
						err := ec2Client.TerminateInstance(instanceID)
						if err != nil {
							fmt.Printf("Failed to terminate instance %s: %v\n", instanceID, err)
							continue
						}
						fmt.Printf("Instance %s termination initiated successfully.\n", instanceID)
					}
				} else {
					fmt.Println("Instance termination cancelled.")
				}

				return nil
			}

			// Handle single instance case
			if len(instances) == 1 {
				instance := instances[0]
				fmt.Println("Found one instance with the specified project tag:")
				fmt.Println(ec2Client.GetInstanceDetails(instance))

				// Ask for confirmation
				fmt.Print("Do you want to terminate this instance? (yes/no): ")
				var response string
				fmt.Scanln(&response)

				if strings.ToLower(response) == "yes" {
					// Terminate the instance
					instanceID := *instance.InstanceId
					err := ec2Client.TerminateInstance(instanceID)
					if err != nil {
						return fmt.Errorf("failed to terminate instance: %v", err)
					}
					fmt.Println("Instance termination initiated successfully.")
				} else {
					fmt.Println("Instance termination cancelled.")
				}

				return nil
			}

			// Multiple instances found and no flags specified - interactive selection
			fmt.Printf("More than one instance matches the `project=%s` tag:\n\n", projectTag)
			for _, instance := range instances {
				fmt.Println(ec2Client.GetInstanceDetails(instance))
				fmt.Println("---")
			}

			// List instance IDs for selection
			fmt.Println("Which one would you like to terminate:")
			for i, instance := range instances {
				fmt.Printf("%d) %s\n", i+1, *instance.InstanceId)
			}

			// Get user selection
			fmt.Print("\nPlease choose one (or simply Enter to exit): ")
			var selection string
			fmt.Scanln(&selection)

			// Handle empty input (exit)
			if selection == "" {
				fmt.Println("No instance selected. Exiting.")
				return nil
			}

			// Parse selection
			var selectionNum int
			_, err = fmt.Sscanf(selection, "%d", &selectionNum)
			if err != nil || selectionNum < 1 || selectionNum > len(instances) {
				return fmt.Errorf("invalid selection: %s", selection)
			}

			// Get selected instance
			selectedInstance := instances[selectionNum-1]
			selectedInstanceID := *selectedInstance.InstanceId
			fmt.Printf("You selected instance %s\n", selectedInstanceID)

			// Ask for confirmation
			fmt.Print("Do you want to terminate this instance? (yes/no): ")
			var response string
			fmt.Scanln(&response)

			if strings.ToLower(response) == "yes" {
				// Terminate the instance
				err := ec2Client.TerminateInstance(selectedInstanceID)
				if err != nil {
					return fmt.Errorf("failed to terminate instance: %v", err)
				}
				fmt.Println("Instance termination initiated successfully.")
			} else {
				fmt.Println("Instance termination cancelled.")
			}

			return nil
		},
	}

	// Add teardown-specific flags
	teardownCmd.Flags().String("instance", "", "Instance ID to terminate (must match project tag)")
	teardownCmd.Flags().Bool("all", false, "Terminate all instances with matching project tag")

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
