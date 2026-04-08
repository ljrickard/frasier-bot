package embeddings

import (
	"context"
	"fmt"
	"log"
	"os"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/types/known/structpb"
)

// Preflight validates that all required configuration is present before
// the scraper starts doing any real work.
func Preflight() error {
	if os.Getenv("GOOGLE_CLOUD_PROJECT") == "" {
		return fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}
	if os.Getenv("GOOGLE_CLOUD_LOCATION") == "" {
		log.Println("GOOGLE_CLOUD_LOCATION not set, will default to europe-west2")
	}
	return nil
}

// GenerateEmbedding calls the Vertex AI text-embedding model and returns a
// float32 slice for the given text.
func GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "europe-west2"
	}

	endpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", location)

	client, err := aiplatform.NewPredictionClient(ctx, option.WithEndpoint(endpoint))
	if err != nil {
		return nil, fmt.Errorf("failed to create prediction client: %w", err)
	}
	defer client.Close()

	model := "text-embedding-004"
	resourceName := fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", project, location, model)

	instance, err := structpb.NewValue(map[string]interface{}{
		"content":   text,
		"task_type": "RETRIEVAL_DOCUMENT",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create instance value: %w", err)
	}

	req := &aiplatformpb.PredictRequest{
		Endpoint:  resourceName,
		Instances: []*structpb.Value{instance},
	}

	resp, err := client.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to predict embedding: %w", err)
	}

	if len(resp.Predictions) == 0 {
		return nil, fmt.Errorf("no predictions returned from embedding model")
	}

	prediction := resp.Predictions[0].GetStructValue()
	if prediction == nil {
		return nil, fmt.Errorf("prediction is not a struct")
	}

	embeddingsField, ok := prediction.Fields["embeddings"]
	if !ok {
		return nil, fmt.Errorf("no 'embeddings' field in prediction response")
	}

	valuesField, ok := embeddingsField.GetStructValue().Fields["values"]
	if !ok {
		return nil, fmt.Errorf("no 'values' field in embeddings")
	}

	listValues := valuesField.GetListValue().GetValues()
	result := make([]float32, len(listValues))
	for i, v := range listValues {
		result[i] = float32(v.GetNumberValue())
	}

	return result, nil
}

// GenerateQueryEmbedding calls the Vertex AI text-embedding model with
// task_type=RETRIEVAL_QUERY, which is optimised for search queries.
func GenerateQueryEmbedding(ctx context.Context, text string) ([]float32, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("GOOGLE_CLOUD_PROJECT environment variable is not set")
	}

	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "europe-west2"
	}

	endpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", location)

	client, err := aiplatform.NewPredictionClient(ctx, option.WithEndpoint(endpoint))
	if err != nil {
		return nil, fmt.Errorf("failed to create prediction client: %w", err)
	}
	defer client.Close()

	model := "text-embedding-004"
	resourceName := fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", project, location, model)

	instance, err := structpb.NewValue(map[string]interface{}{
		"content":   text,
		"task_type": "RETRIEVAL_QUERY",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create instance value: %w", err)
	}

	req := &aiplatformpb.PredictRequest{
		Endpoint:  resourceName,
		Instances: []*structpb.Value{instance},
	}

	resp, err := client.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to predict embedding: %w", err)
	}

	if len(resp.Predictions) == 0 {
		return nil, fmt.Errorf("no predictions returned from embedding model")
	}

	prediction := resp.Predictions[0].GetStructValue()
	if prediction == nil {
		return nil, fmt.Errorf("prediction is not a struct")
	}

	embeddingsField, ok := prediction.Fields["embeddings"]
	if !ok {
		return nil, fmt.Errorf("no 'embeddings' field in prediction response")
	}

	valuesField, ok := embeddingsField.GetStructValue().Fields["values"]
	if !ok {
		return nil, fmt.Errorf("no 'values' field in embeddings")
	}

	listValues := valuesField.GetListValue().GetValues()
	result := make([]float32, len(listValues))
	for i, v := range listValues {
		result[i] = float32(v.GetNumberValue())
	}

	return result, nil
}
