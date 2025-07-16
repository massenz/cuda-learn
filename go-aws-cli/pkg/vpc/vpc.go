package vpc

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	"github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/cuda-learn/go-aws-cli/pkg/common"
)

// VPCClient handles VPC-related operations
type VPCClient struct {
	client *ec2.Client
}

// NewVPCClient creates a new VPC client
func NewVPCClient(cfg aws.Config) *VPCClient {
	return &VPCClient{
		client: ec2.NewFromConfig(cfg),
	}
}

// SetupVPC creates or finds a VPC with the specified tag
func (v *VPCClient) SetupVPC(projectTag, vpcCidr, subnetCidr string) (string, string, error) {
	// Check if VPC with tag exists
	vpcID, err := v.findVPCByTag(projectTag)
	if err != nil {
		return "", "", fmt.Errorf("error finding VPC: %w", err)
	}

	// If VPC exists, find subnet
	if vpcID != "" {
		common.LogInfo("Found existing VPC: %s", vpcID)

		// Find subnet in the VPC
		subnetID, err := v.findSubnetByTag(vpcID, projectTag)
		if err != nil {
			return "", "", fmt.Errorf("error finding subnet: %w", err)
		}

		if subnetID == "" {
			return "", "", fmt.Errorf("no subnet found in VPC %s with tag project=%s", vpcID, projectTag)
		}

		common.LogInfo("Found existing Subnet: %s", subnetID)
		return vpcID, subnetID, nil
	}

	// Create new VPC
	common.LogInfo("No existing VPC found. Creating new one...")
	vpcID, err = v.createVPC(projectTag, vpcCidr)
	if err != nil {
		return "", "", fmt.Errorf("error creating VPC: %w", err)
	}

	// Create subnet
	subnetID, err := v.createSubnet(vpcID, projectTag, subnetCidr)
	if err != nil {
		return "", "", fmt.Errorf("error creating subnet: %w", err)
	}

	// Create and attach internet gateway
	igwID, err := v.createAndAttachInternetGateway(vpcID)
	if err != nil {
		return "", "", fmt.Errorf("error creating/attaching internet gateway: %w", err)
	}

	// Create route table and routes
	_, err = v.createRouteTable(vpcID, subnetID, igwID)
	if err != nil {
		return "", "", fmt.Errorf("error creating route table: %w", err)
	}

	// Enable auto-assign public IP
	err = v.enableAutoAssignPublicIP(subnetID)
	if err != nil {
		return "", "", fmt.Errorf("error enabling auto-assign public IP: %w", err)
	}

	return vpcID, subnetID, nil
}

// findVPCByTag finds a VPC with the specified tag
func (v *VPCClient) findVPCByTag(projectTag string) (string, error) {
	input := &ec2.DescribeVpcsInput{
		Filters: []types.Filter{
			{
				Name:   aws.String("tag:project"),
				Values: []string{projectTag},
			},
		},
	}

	resp, err := v.client.DescribeVpcs(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to describe VPCs: %w", err)
	}

	if len(resp.Vpcs) > 0 {
		return *resp.Vpcs[0].VpcId, nil
	}

	return "", nil
}

// findSubnetByTag finds a subnet in the specified VPC with the specified tag
func (v *VPCClient) findSubnetByTag(vpcID, projectTag string) (string, error) {
	input := &ec2.DescribeSubnetsInput{
		Filters: []types.Filter{
			{
				Name:   aws.String("vpc-id"),
				Values: []string{vpcID},
			},
			{
				Name:   aws.String("tag:project"),
				Values: []string{projectTag},
			},
		},
	}

	resp, err := v.client.DescribeSubnets(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to describe subnets: %w", err)
	}

	if len(resp.Subnets) > 0 {
		return *resp.Subnets[0].SubnetId, nil
	}

	return "", nil
}

// createVPC creates a new VPC with the specified CIDR block
func (v *VPCClient) createVPC(projectTag, cidrBlock string) (string, error) {
	input := &ec2.CreateVpcInput{
		CidrBlock: aws.String(cidrBlock),
	}

	resp, err := v.client.CreateVpc(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to create VPC: %w", err)
	}

	vpcID := *resp.Vpc.VpcId

	// Add tags
	tagInput := &ec2.CreateTagsInput{
		Resources: []string{vpcID},
		Tags: []types.Tag{
			{
				Key:   aws.String("project"),
				Value: aws.String(projectTag),
			},
		},
	}

	_, err = v.client.CreateTags(context.TODO(), tagInput)
	if err != nil {
		return "", fmt.Errorf("failed to tag VPC: %w", err)
	}

	common.LogSuccess("Created VPC: %s", vpcID)
	return vpcID, nil
}

// createSubnet creates a new subnet in the specified VPC
func (v *VPCClient) createSubnet(vpcID, projectTag, cidrBlock string) (string, error) {
	input := &ec2.CreateSubnetInput{
		VpcId:            aws.String(vpcID),
		CidrBlock:        aws.String(cidrBlock),
		AvailabilityZone: aws.String("us-west-2a"),
	}

	resp, err := v.client.CreateSubnet(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to create subnet: %w", err)
	}

	subnetID := *resp.Subnet.SubnetId

	// Add tags
	tagInput := &ec2.CreateTagsInput{
		Resources: []string{subnetID},
		Tags: []types.Tag{
			{
				Key:   aws.String("project"),
				Value: aws.String(projectTag),
			},
		},
	}

	_, err = v.client.CreateTags(context.TODO(), tagInput)
	if err != nil {
		return "", fmt.Errorf("failed to tag subnet: %w", err)
	}

	common.LogSuccess("Created Subnet: %s", subnetID)
	return subnetID, nil
}

// createAndAttachInternetGateway creates a new internet gateway and attaches it to the VPC
func (v *VPCClient) createAndAttachInternetGateway(vpcID string) (string, error) {
	// Create internet gateway
	igwResp, err := v.client.CreateInternetGateway(context.TODO(), &ec2.CreateInternetGatewayInput{})
	if err != nil {
		return "", fmt.Errorf("failed to create internet gateway: %w", err)
	}

	igwID := *igwResp.InternetGateway.InternetGatewayId

	// Attach internet gateway to VPC
	_, err = v.client.AttachInternetGateway(context.TODO(), &ec2.AttachInternetGatewayInput{
		InternetGatewayId: aws.String(igwID),
		VpcId:             aws.String(vpcID),
	})
	if err != nil {
		return "", fmt.Errorf("failed to attach internet gateway: %w", err)
	}

	common.LogSuccess("Created and attached Internet Gateway: %s", igwID)
	return igwID, nil
}

// createRouteTable creates a new route table and associates it with the subnet
func (v *VPCClient) createRouteTable(vpcID, subnetID, igwID string) (string, error) {
	// Create route table
	rtbResp, err := v.client.CreateRouteTable(context.TODO(), &ec2.CreateRouteTableInput{
		VpcId: aws.String(vpcID),
	})
	if err != nil {
		return "", fmt.Errorf("failed to create route table: %w", err)
	}

	rtbID := *rtbResp.RouteTable.RouteTableId

	// Create route to internet gateway
	_, err = v.client.CreateRoute(context.TODO(), &ec2.CreateRouteInput{
		RouteTableId:         aws.String(rtbID),
		DestinationCidrBlock: aws.String("0.0.0.0/0"),
		GatewayId:            aws.String(igwID),
	})
	if err != nil {
		return "", fmt.Errorf("failed to create route: %w", err)
	}

	// Associate route table with subnet
	_, err = v.client.AssociateRouteTable(context.TODO(), &ec2.AssociateRouteTableInput{
		RouteTableId: aws.String(rtbID),
		SubnetId:     aws.String(subnetID),
	})
	if err != nil {
		return "", fmt.Errorf("failed to associate route table: %w", err)
	}

	common.LogSuccess("Created Route Table: %s and associated with Subnet: %s", rtbID, subnetID)
	return rtbID, nil
}

// enableAutoAssignPublicIP enables auto-assign public IP for the subnet
func (v *VPCClient) enableAutoAssignPublicIP(subnetID string) error {
	_, err := v.client.ModifySubnetAttribute(context.TODO(), &ec2.ModifySubnetAttributeInput{
		SubnetId:            aws.String(subnetID),
		MapPublicIpOnLaunch: &types.AttributeBooleanValue{Value: aws.Bool(true)},
	})
	if err != nil {
		return fmt.Errorf("failed to enable auto-assign public IP: %w", err)
	}

	common.LogSuccess("Enabled auto-assign public IP for Subnet: %s", subnetID)
	return nil
}

// CreateSecurityGroup creates a security group for SSH access
func (v *VPCClient) CreateSecurityGroup(vpcID, projectTag string) (string, error) {
	// Check if security group already exists
	sgID, err := v.findSecurityGroupByName(vpcID, "ssh-sg")
	if err != nil {
		return "", fmt.Errorf("error finding security group: %w", err)
	}

	if sgID != "" {
		common.LogInfo("Found existing Security Group: %s", sgID)
		return sgID, nil
	}

	// Create security group
	common.LogInfo("Creating Security Group and enabling SSH access...")
	sgResp, err := v.client.CreateSecurityGroup(context.TODO(), &ec2.CreateSecurityGroupInput{
		GroupName:   aws.String("ssh-sg"),
		Description: aws.String("SSH access"),
		VpcId:       aws.String(vpcID),
	})
	if err != nil {
		return "", fmt.Errorf("failed to create security group: %w", err)
	}

	sgID = *sgResp.GroupId

	// Add SSH ingress rule
	_, err = v.client.AuthorizeSecurityGroupIngress(context.TODO(), &ec2.AuthorizeSecurityGroupIngressInput{
		GroupId:    aws.String(sgID),
		IpProtocol: aws.String("tcp"),
		FromPort:   aws.Int32(22),
		ToPort:     aws.Int32(22),
		CidrIp:     aws.String("0.0.0.0/0"),
	})
	if err != nil {
		return "", fmt.Errorf("failed to authorize security group ingress: %w", err)
	}

	common.LogSuccess("Created SG for SSH Access: %s", sgID)
	return sgID, nil
}

// findSecurityGroupByName finds a security group by name in the specified VPC
func (v *VPCClient) findSecurityGroupByName(vpcID, groupName string) (string, error) {
	input := &ec2.DescribeSecurityGroupsInput{
		Filters: []types.Filter{
			{
				Name:   aws.String("vpc-id"),
				Values: []string{vpcID},
			},
			{
				Name:   aws.String("group-name"),
				Values: []string{groupName},
			},
		},
	}

	resp, err := v.client.DescribeSecurityGroups(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to describe security groups: %w", err)
	}

	if len(resp.SecurityGroups) > 0 {
		return *resp.SecurityGroups[0].GroupId, nil
	}

	return "", nil
}
