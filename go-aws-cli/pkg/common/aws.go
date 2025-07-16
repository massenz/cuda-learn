package common

import (
	"context"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ec2/types"
)

// InitAWSConfig initializes and returns an AWS configuration for the specified region
func InitAWSConfig(region string) (aws.Config, error) {
	log.Printf("Initializing AWS configuration for region: %s", region)

	// Load AWS configuration
	cfg, err := config.LoadDefaultConfig(context.TODO(),
		config.WithRegion(region),
	)
	if err != nil {
		return aws.Config{}, fmt.Errorf("unable to load AWS SDK config: %w", err)
	}

	return cfg, nil
}

// CreateTagSpecifications creates AWS tag specifications for resources
func CreateTagSpecifications(resourceType string, projectTag string, additionalTags map[string]string) []types.Tag {
	// Create base tags
	tags := []types.Tag{
		{
			Key:   aws.String("project"),
			Value: aws.String(projectTag),
		},
	}

	// Add additional tags if provided
	for key, value := range additionalTags {
		tags = append(tags, types.Tag{
			Key:   aws.String(key),
			Value: aws.String(value),
		})
	}

	return tags
}

// LogInfo logs an informational message
func LogInfo(format string, args ...interface{}) {
	log.Printf("[INFO] "+format, args...)
}

// LogError logs an error message
func LogError(format string, args ...interface{}) {
	log.Printf("[ERROR] "+format, args...)
}

// LogSuccess logs a success message
func LogSuccess(format string, args ...interface{}) {
	log.Printf("[SUCCESS] "+format, args...)
}
