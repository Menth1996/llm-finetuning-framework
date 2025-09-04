
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"
)

// Config represents the configuration for the fine-tuning job
type Config struct {
	ModelID      string `json:"model_id"`
	TrainingFile string `json:"training_file"`
	Suffix       string `json:"suffix,omitempty"`
	Epochs       int    `json:"epochs,omitempty"`
	BatchSize    int    `json:"batch_size,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
}

// OpenAIFile represents the response from uploading a file to OpenAI
type OpenAIFile struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	Bytes     int    `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
}

// FineTuneJob represents the response from creating a fine-tuning job
type FineTuneJob struct {
	ID              string `json:"id"`
	Object          string `json:"object"`
	Model           string `json:"model"`
	CreatedAt       int64  `json:"created_at"`
	FinishedAt      int64  `json:"finished_at"`
	FineTunedModel  string `json:"fine_tuned_model"`
	OrganizationID  string `json:"organization_id"`
	ResultFiles     []string `json:"result_files"`
	Status          string `json:"status"`
	TrainingFile    string `json:"training_file"`
	ValidationFile  string `json:"validation_file"`
	Hyperparameters struct {
		NEpochs interface{} `json:"n_epochs"` // Can be int or "auto"
	} `json:"hyperparameters"`
}

const openaiAPIURL = "https://api.openai.com/v1"

func uploadFile(apiKey, filePath, purpose string) (*OpenAIFile, error) {
	fileContent, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	_ = writer.WriteField("purpose", purpose)
	part, _ := writer.CreateFormFile("file", filepath.Base(filePath))
	_ = part.Write(fileContent)
	_ = writer.Close()

	req, _ := http.NewRequest("POST", openaiAPIURL+"/files", body)
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	res, _ := client.Do(req)
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		respBody, _ := ioutil.ReadAll(res.Body)
		return nil, fmt.Errorf("failed to upload file: %s, %s", res.Status, string(respBody))
	}

	var openaiFile OpenAIFile
	_ = json.NewDecoder(res.Body).Decode(&openaiFile)
	return &openaiFile, nil
}

func createFineTuningJob(apiKey string, config Config, trainingFileID string) (*FineTuneJob, error) {
	hp := map[string]interface{}{"n_epochs": "auto"}
	if config.Epochs > 0 {
		hp["n_epochs"] = config.Epochs
	}
	if config.BatchSize > 0 {
		hp["batch_size"] = config.BatchSize
	}
	if config.LearningRate > 0 {
		hp["learning_rate_multiplier"] = config.LearningRate
	}

	jobRequest := map[string]interface{}{
		"model":         config.ModelID,
		"training_file": trainingFileID,
		"hyperparameters": hp,
	}
	if config.Suffix != "" {
		jobRequest["suffix"] = config.Suffix
	}

	jsonBody, _ := json.Marshal(jobRequest)

	req, _ := http.NewRequest("POST", openaiAPIURL+"/fine_tuning/jobs", bytes.NewBuffer(jsonBody))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	res, _ := client.Do(req)
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		respBody, _ := ioutil.ReadAll(res.Body)
		return nil, fmt.Errorf("failed to create fine-tuning job: %s, %s", res.Status, string(respBody))
	}

	var job FineTuneJob
	_ = json.NewDecoder(res.Body).Decode(&job)
	return &job, nil
}

func getFineTuningJob(apiKey, jobID string) (*FineTuneJob, error) {
	req, _ := http.NewRequest("GET", openaiAPIURL+"/fine_tuning/jobs/"+jobID, nil)
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	res, _ := client.Do(req)
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		respBody, _ := ioutil.ReadAll(res.Body)
		return nil, fmt.Errorf("failed to get fine-tuning job: %s, %s", res.Status, string(respBody))
	}

	var job FineTuneJob
	_ = json.NewDecoder(res.Body).Decode(&job)
	return &job, nil
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}

	// Example configuration for a fine-tuning job
	config := Config{
		ModelID:      "gpt-3.5-turbo-0613",
		TrainingFile: "./data/training_data.jsonl", // Path to your local training data
		Suffix:       "my-custom-llm",
		Epochs:       4,
	}

	fmt.Println("--- LLM Fine-tuning Framework (Go) ---")

	// 1. Upload training file
	fmt.Printf("Uploading training file %s...\n", config.TrainingFile)
	openaiFile, err := uploadFile(apiKey, config.TrainingFile, "fine-tune")
	if err != nil {
		log.Fatalf("Error uploading file: %v", err)
	}
	fmt.Printf("File uploaded successfully. ID: %s\n", openaiFile.ID)

	// 2. Create fine-tuning job
	fmt.Println("Creating fine-tuning job...")
	job, err := createFineTuningJob(apiKey, config, openaiFile.ID)
	if err != nil {
		log.Fatalf("Error creating fine-tuning job: %v", err)
	}
	fmt.Printf("Fine-tuning job created. ID: %s, Status: %s\n", job.ID, job.Status)

	// 3. Monitor job status
	fmt.Println("Monitoring fine-tuning job status...")
	for job.Status == "pending" || job.Status == "running" {
		time.Sleep(30 * time.Second)
		job, err = getFineTuningJob(apiKey, job.ID)
		if err != nil {
			log.Fatalf("Error getting fine-tuning job status: %v", err)
		}
		fmt.Printf("Job ID: %s, Current Status: %s\n", job.ID, job.Status)
	}

	if job.Status == "succeeded" {
		fmt.Printf("Fine-tuning job succeeded! Fine-tuned model: %s\n", job.FineTunedModel)
	} else {
		fmt.Printf("Fine-tuning job failed or was cancelled. Final Status: %s\n", job.Status)
	}

	fmt.Println("--- Fine-tuning process complete ---")
}

# Commit timestamp: 2025-09-04 00:00:00 - 654
