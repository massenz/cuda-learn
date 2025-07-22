/*
 * Copyright (c) 2025 AlertAvert.com.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Author: Marco Massenzio (marco@alertavert.com)
 */

package iam

import (
	"context"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/iam"
	"github.com/aws/aws-sdk-go-v2/service/iam/types"
	"github.com/cuda-learn/go-aws-cli/pkg/common"
)

// IAMClient handles IAM-related operations
type IAMClient struct {
	iamClient *iam.Client
}

// NewIAMClient creates a new IAM client
func NewIAMClient(cfg aws.Config) *IAMClient {
	return &IAMClient{
		iamClient: iam.NewFromConfig(cfg),
	}
}

// EC2RoleName returns the standard role name for EC2 instances
func EC2RoleName(projectTag string) string {
	return fmt.Sprintf("%s-ec2-role", projectTag)
}

// EC2InstanceProfileName returns the standard instance profile name for EC2 instances
func EC2InstanceProfileName(projectTag string) string {
	return fmt.Sprintf("%s-profile", projectTag)
}

// SetupEC2Role creates an IAM role and instance profile for EC2 instances if they don't exist
// Returns the instance profile ARN
func (i *IAMClient) SetupEC2Role(projectTag string) (string, error) {
	roleName := EC2RoleName(projectTag)
	profileName := EC2InstanceProfileName(projectTag)

	// Check if role exists
	roleArn, err := i.findRoleByName(roleName)
	if err != nil {
		return "", fmt.Errorf("error finding role: %w", err)
	}

	// Create role if it doesn't exist
	if roleArn == "" {
		common.LogInfo("Creating IAM role: %s", roleName)
		roleArn, err = i.createEC2Role(roleName, projectTag)
		if err != nil {
			return "", fmt.Errorf("failed to create role: %w", err)
		}
		common.LogSuccess("Created IAM role: %s with ARN: %s", roleName, roleArn)

		// Attach policies to the role
		err = i.attachPolicyToRole(roleName, "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
		if err != nil {
			return "", fmt.Errorf("failed to attach SSM policy: %w", err)
		}

		err = i.attachPolicyToRole(roleName, "arn:aws:iam::aws:policy/SecretsManagerReadWrite")
		if err != nil {
			return "", fmt.Errorf("failed to attach Secrets Manager policy: %w", err)
		}
		common.LogInfo("Attached policies to role: %s", roleName)
	} else {
		common.LogInfo("Found existing IAM role: %s with ARN: %s", roleName, roleArn)
	}

	// Check if instance profile exists
	profileArn, err := i.findInstanceProfileByName(profileName)
	if err != nil {
		return "", fmt.Errorf("error finding instance profile: %w", err)
	}

	// Create instance profile if it doesn't exist
	if profileArn == "" {
		common.LogInfo("Creating IAM instance profile: %s", profileName)
		profileArn, err = i.createInstanceProfile(profileName, roleName)
		if err != nil {
			return "", fmt.Errorf("failed to create instance profile: %w", err)
		}
		common.LogSuccess("Created IAM instance profile: %s with ARN: %s", profileName, profileArn)
	} else {
		common.LogInfo("Found existing IAM instance profile: %s with ARN: %s", profileName, profileArn)
	}

	return profileArn, nil
}

// findRoleByName finds an IAM role by name
// Returns the role ARN if found, empty string if not found
func (i *IAMClient) findRoleByName(roleName string) (string, error) {
	input := &iam.GetRoleInput{
		RoleName: aws.String(roleName),
	}

	resp, err := i.iamClient.GetRole(context.TODO(), input)
	if err != nil {
		// If the role doesn't exist, AWS returns an error
		if strings.Contains(err.Error(), "NoSuchEntity") {
			return "", nil
		}
		return "", fmt.Errorf("failed to get role: %w", err)
	}

	return *resp.Role.Arn, nil
}

// findInstanceProfileByName finds an IAM instance profile by name
// Returns the instance profile ARN if found, empty string if not found
func (i *IAMClient) findInstanceProfileByName(profileName string) (string, error) {
	input := &iam.GetInstanceProfileInput{
		InstanceProfileName: aws.String(profileName),
	}

	resp, err := i.iamClient.GetInstanceProfile(context.TODO(), input)
	if err != nil {
		// If the instance profile doesn't exist, AWS returns an error
		if strings.Contains(err.Error(), "NoSuchEntity") {
			return "", nil
		}
		return "", fmt.Errorf("failed to get instance profile: %w", err)
	}

	return *resp.InstanceProfile.Arn, nil
}

// createEC2Role creates an IAM role for EC2 instances
func (i *IAMClient) createEC2Role(roleName string, projectTag string) (string, error) {
	// Trust policy document for EC2
	trustPolicy := `{
		"Version": "2012-10-17",
		"Statement": [
			{
				"Effect": "Allow",
				"Principal": {
					"Service": "ec2.amazonaws.com"
				},
				"Action": "sts:AssumeRole"
			}
		]
	}`

	input := &iam.CreateRoleInput{
		RoleName:                 aws.String(roleName),
		AssumeRolePolicyDocument: aws.String(trustPolicy),
		Description:              aws.String(fmt.Sprintf("Role for EC2 instances in %s project", projectTag)),
		Tags: []types.Tag{
			{
				Key:   aws.String("project"),
				Value: aws.String(projectTag),
			},
		},
	}

	resp, err := i.iamClient.CreateRole(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to create role: %w", err)
	}

	return *resp.Role.Arn, nil
}

// attachPolicyToRole attaches a policy to an IAM role
func (i *IAMClient) attachPolicyToRole(roleName string, policyArn string) error {
	input := &iam.AttachRolePolicyInput{
		RoleName:  aws.String(roleName),
		PolicyArn: aws.String(policyArn),
	}

	_, err := i.iamClient.AttachRolePolicy(context.TODO(), input)
	if err != nil {
		return fmt.Errorf("failed to attach policy: %w", err)
	}

	return nil
}

// createInstanceProfile creates an IAM instance profile and attaches the role
func (i *IAMClient) createInstanceProfile(profileName string, roleName string) (string, error) {
	// Create instance profile
	createInput := &iam.CreateInstanceProfileInput{
		InstanceProfileName: aws.String(profileName),
	}

	createResp, err := i.iamClient.CreateInstanceProfile(context.TODO(), createInput)
	if err != nil {
		return "", fmt.Errorf("failed to create instance profile: %w", err)
	}

	// Add role to instance profile
	addRoleInput := &iam.AddRoleToInstanceProfileInput{
		InstanceProfileName: aws.String(profileName),
		RoleName:            aws.String(roleName),
	}

	_, err = i.iamClient.AddRoleToInstanceProfile(context.TODO(), addRoleInput)
	if err != nil {
		return "", fmt.Errorf("failed to add role to instance profile: %w", err)
	}

	return *createResp.InstanceProfile.Arn, nil
}
