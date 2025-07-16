package ec2

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	"github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	"github.com/cuda-learn/go-aws-cli/pkg/common"
	"github.com/cuda-learn/go-aws-cli/pkg/vpc"
)

// EC2Client handles EC2-related operations
type EC2Client struct {
	ec2Client     *ec2.Client
	secretsClient *secretsmanager.Client
	vpcClient     *vpc.VPCClient
}

// NewEC2Client creates a new EC2 client
func NewEC2Client(cfg aws.Config) *EC2Client {
	return &EC2Client{
		ec2Client:     ec2.NewFromConfig(cfg),
		secretsClient: secretsmanager.NewFromConfig(cfg),
		vpcClient:     vpc.NewVPCClient(cfg),
	}
}

// SetupEC2 creates an EC2 instance
func (e *EC2Client) SetupEC2(projectTag, vpcID, subnetID, keyName, instanceType string) (string, string, error) {
	// Create or find security group
	sgID, err := e.vpcClient.CreateSecurityGroup(vpcID, projectTag)
	if err != nil {
		return "", "", fmt.Errorf("failed to create security group: %w", err)
	}
	common.LogInfo("SSH Access enabled (Security Group: %s)", sgID)

	// Create or find key pair
	err = e.setupKeyPair(keyName)
	if err != nil {
		return "", "", fmt.Errorf("failed to setup key pair: %w", err)
	}
	common.LogInfo("Key pair in private/%s.pem", keyName)

	// Find AMI
	amiID, amiName, err := e.findLatestAMI()
	if err != nil {
		return "", "", fmt.Errorf("failed to find AMI: %w", err)
	}
	common.LogInfo("Reserving an EC2 Instance (%s), AMI: %s (%s)", instanceType, amiID, amiName)

	// Launch instance
	instanceID, err := e.launchInstance(amiID, instanceType, keyName, sgID, subnetID)
	if err != nil {
		return "", "", fmt.Errorf("failed to launch instance: %w", err)
	}
	common.LogInfo("Launched instance: %s", instanceID)

	// Wait for instance to be running
	err = e.waitForInstanceRunning(instanceID)
	if err != nil {
		return "", "", fmt.Errorf("error waiting for instance to be running: %w", err)
	}
	common.LogSuccess("Instance %s is now running.", instanceID)

	// Get public IP
	publicIP, err := e.getInstancePublicIP(instanceID)
	if err != nil {
		return "", "", fmt.Errorf("failed to get instance public IP: %w", err)
	}
	common.LogInfo("Public IP: %s", publicIP)

	return instanceID, publicIP, nil
}

// setupKeyPair creates or finds a key pair
func (e *EC2Client) setupKeyPair(keyName string) error {
	// Check if key pair exists
	keyID, err := e.findKeyPair(keyName)
	if err != nil {
		return fmt.Errorf("error finding key pair: %w", err)
	}

	if keyID != "" {
		common.LogInfo("Found existing key pair: %s", keyID)
		return nil
	}

	// Create private directory if it doesn't exist
	err = os.MkdirAll("private", 0755)
	if err != nil {
		return fmt.Errorf("failed to create private directory: %w", err)
	}

	// Generate key pair
	privateKey, publicKey, err := e.generateKeyPair()
	if err != nil {
		return fmt.Errorf("failed to generate key pair: %w", err)
	}

	// Save private key to file
	keyPath := filepath.Join("private", keyName+".pem")
	err = e.savePrivateKey(privateKey, keyPath)
	if err != nil {
		return fmt.Errorf("failed to save private key: %w", err)
	}

	// Import key pair to AWS
	keyID, err = e.importKeyPair(keyName, publicKey)
	if err != nil {
		return fmt.Errorf("failed to import key pair: %w", err)
	}

	// Store private key in SecretsManager
	err = e.storeKeyInSecretsManager(keyName, privateKey)
	if err != nil {
		return fmt.Errorf("failed to store key in SecretsManager: %w", err)
	}

	common.LogSuccess("Created SSH key %s (%s)", keyName, keyID)
	return nil
}

// findKeyPair finds a key pair by name
func (e *EC2Client) findKeyPair(keyName string) (string, error) {
	input := &ec2.DescribeKeyPairsInput{
		KeyNames: []string{keyName},
	}

	resp, err := e.ec2Client.DescribeKeyPairs(context.TODO(), input)
	if err != nil {
		// If the key doesn't exist, AWS returns an error
		if strings.Contains(err.Error(), "InvalidKeyPair.NotFound") {
			return "", nil
		}
		return "", fmt.Errorf("failed to describe key pairs: %w", err)
	}

	if len(resp.KeyPairs) > 0 {
		return *resp.KeyPairs[0].KeyPairId, nil
	}

	return "", nil
}

// generateKeyPair generates a new RSA key pair
func (e *EC2Client) generateKeyPair() (*rsa.PrivateKey, []byte, error) {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	// Extract public key
	publicKey, err := x509.MarshalPKIXPublicKey(&privateKey.PublicKey)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal public key: %w", err)
	}

	return privateKey, publicKey, nil
}

// savePrivateKey saves the private key to a file
func (e *EC2Client) savePrivateKey(privateKey *rsa.PrivateKey, keyPath string) error {
	// Create PEM block
	privateKeyPEM := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(privateKey),
	}

	// Create file
	file, err := os.OpenFile(keyPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("failed to create private key file: %w", err)
	}
	defer file.Close()

	// Write PEM block
	err = pem.Encode(file, privateKeyPEM)
	if err != nil {
		return fmt.Errorf("failed to write private key: %w", err)
	}

	return nil
}

// importKeyPair imports a key pair to AWS
func (e *EC2Client) importKeyPair(keyName string, publicKey []byte) (string, error) {
	input := &ec2.ImportKeyPairInput{
		KeyName:           aws.String(keyName),
		PublicKeyMaterial: publicKey,
	}

	resp, err := e.ec2Client.ImportKeyPair(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to import key pair: %w", err)
	}

	return *resp.KeyPairId, nil
}

// storeKeyInSecretsManager stores the private key in AWS SecretsManager
func (e *EC2Client) storeKeyInSecretsManager(keyName string, privateKey *rsa.PrivateKey) error {
	// Create PEM block
	privateKeyPEM := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(privateKey),
	}

	// Encode to PEM format
	pemBytes := pem.EncodeToMemory(privateKeyPEM)

	// Create or update secret
	secretName := fmt.Sprintf("ssh-key-%s", keyName)

	// Check if secret exists
	_, err := e.secretsClient.DescribeSecret(context.TODO(), &secretsmanager.DescribeSecretInput{
		SecretId: aws.String(secretName),
	})

	if err != nil {
		// Create new secret
		_, err = e.secretsClient.CreateSecret(context.TODO(), &secretsmanager.CreateSecretInput{
			Name:         aws.String(secretName),
			SecretString: aws.String(string(pemBytes)),
			Description:  aws.String(fmt.Sprintf("SSH private key for %s", keyName)),
		})
		if err != nil {
			return fmt.Errorf("failed to create secret: %w", err)
		}
	} else {
		// Update existing secret
		_, err = e.secretsClient.PutSecretValue(context.TODO(), &secretsmanager.PutSecretValueInput{
			SecretId:     aws.String(secretName),
			SecretString: aws.String(string(pemBytes)),
		})
		if err != nil {
			return fmt.Errorf("failed to update secret: %w", err)
		}
	}

	common.LogSuccess("Stored SSH key in SecretsManager: %s", secretName)
	return nil
}

// findLatestAMI finds the latest GPU-enabled PyTorch AMI
func (e *EC2Client) findLatestAMI() (string, string, error) {
	input := &ec2.DescribeImagesInput{
		Owners: []string{"amazon"},
		Filters: []types.Filter{
			{
				Name:   aws.String("name"),
				Values: []string{"*GPU*PyTorch*2*Ubuntu*"},
			},
			{
				Name:   aws.String("architecture"),
				Values: []string{"x86_64"},
			},
		},
	}

	resp, err := e.ec2Client.DescribeImages(context.TODO(), input)
	if err != nil {
		return "", "", fmt.Errorf("failed to describe images: %w", err)
	}

	if len(resp.Images) == 0 {
		return "", "", fmt.Errorf("no matching AMIs found")
	}

	// Sort by creation date (newest first)
	sort.Slice(resp.Images, func(i, j int) bool {
		iTime, _ := time.Parse(time.RFC3339, *resp.Images[i].CreationDate)
		jTime, _ := time.Parse(time.RFC3339, *resp.Images[j].CreationDate)
		return iTime.After(jTime)
	})

	// Get the newest AMI
	amiID := *resp.Images[0].ImageId
	amiName := *resp.Images[0].Name

	return amiID, amiName, nil
}

// launchInstance launches a new EC2 instance
func (e *EC2Client) launchInstance(amiID, instanceType, keyName, sgID, subnetID string) (string, error) {
	input := &ec2.RunInstancesInput{
		ImageId:          aws.String(amiID),
		InstanceType:     types.InstanceType(instanceType),
		MinCount:         aws.Int32(1),
		MaxCount:         aws.Int32(1),
		KeyName:          aws.String(keyName),
		SecurityGroupIds: []string{sgID},
		SubnetId:         aws.String(subnetID),
	}

	resp, err := e.ec2Client.RunInstances(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to launch instance: %w", err)
	}

	if len(resp.Instances) == 0 {
		return "", fmt.Errorf("no instances launched")
	}

	return *resp.Instances[0].InstanceId, nil
}

// waitForInstanceRunning waits for an instance to be in the running state
func (e *EC2Client) waitForInstanceRunning(instanceID string) error {
	waiter := ec2.NewInstanceRunningWaiter(e.ec2Client)
	return waiter.Wait(context.TODO(), &ec2.DescribeInstancesInput{
		InstanceIds: []string{instanceID},
	}, 5*time.Minute)
}

// getInstancePublicIP gets the public IP of an instance
func (e *EC2Client) getInstancePublicIP(instanceID string) (string, error) {
	input := &ec2.DescribeInstancesInput{
		InstanceIds: []string{instanceID},
	}

	resp, err := e.ec2Client.DescribeInstances(context.TODO(), input)
	if err != nil {
		return "", fmt.Errorf("failed to describe instance: %w", err)
	}

	if len(resp.Reservations) == 0 || len(resp.Reservations[0].Instances) == 0 {
		return "", fmt.Errorf("instance not found")
	}

	instance := resp.Reservations[0].Instances[0]
	if instance.PublicIpAddress == nil {
		return "", fmt.Errorf("instance does not have a public IP")
	}

	return *instance.PublicIpAddress, nil
}
