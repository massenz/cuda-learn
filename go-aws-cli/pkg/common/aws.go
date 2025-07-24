/*
 * Copyright (c) 2025 AlertAvert.com.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Author: Marco Massenzio (marco@alertavert.com)
 */

package common

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// VerboseMode controls whether logs are also output to console
var VerboseMode bool

// InitLogger initializes the zerolog logger
// If verbose is true, logs will be output to both console and file
// Otherwise, logs will only be output to file
func InitLogger(verbose bool) error {
	// Set global verbose mode
	VerboseMode = verbose

	// Create logs directory if it doesn't exist
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %w", err)
	}

	logDir := filepath.Join(homeDir, ".cuda-learn", "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return fmt.Errorf("failed to create log directory: %w", err)
	}

	// Open log file
	logFile, err := os.OpenFile(
		filepath.Join(logDir, "cuda-learn.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND,
		0644,
	)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}

	// Configure zerolog
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix

	// Set up multi-writer if verbose mode is enabled
	if verbose {
		// Output to both console and file
		multi := zerolog.MultiLevelWriter(zerolog.ConsoleWriter{Out: os.Stdout}, logFile)
		log.Logger = zerolog.New(multi).With().Timestamp().Logger()
	} else {
		// Output only to file
		log.Logger = zerolog.New(logFile).With().Timestamp().Logger()
	}

	return nil
}

// InitAWSConfig initializes and returns an AWS configuration for the specified region
func InitAWSConfig(region string) (aws.Config, error) {
	log.Info().Str("region", region).Msg("Initializing AWS configuration")

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
// values is a map of key-value pairs to add to the log event
func LogInfo(msg string, values map[string]string) {
	event := log.Info()

	// Add fields if provided
	if values != nil {
		for k, v := range values {
			event = event.Str(k, v)
		}
	}

	// Send the message
	event.Msg(msg)
}

// LogError logs an error message
// err is the error to log
// msg is an optional message to include with the error
func LogError(err error, msg string) {
	log.Error().Err(err).Msg(msg)
}

// LogSuccess logs a success message
// Reuses LogInfo code, adding a {"status", "success"} to the fields map
// and "[SUCCESS]" prefix to the msg
func LogSuccess(msg string, values map[string]string) {
	// Create a new map if values is nil, otherwise make a copy
	successValues := make(map[string]string)
	if values != nil {
		for k, v := range values {
			successValues[k] = v
		}
	}

	// Add status=success to the fields map
	successValues["status"] = "success"

	// Add [SUCCESS] prefix to the message
	successMsg := "[SUCCESS] " + msg

	// Call LogInfo with the modified parameters
	LogInfo(successMsg, successValues)
}

// UserMessage prints a message to stdout for user feedback
// This is used for user-friendly progress messages
func UserMessage(msg string) {
	if !VerboseMode {
		fmt.Printf(".. %s\n", msg)
	}
}
