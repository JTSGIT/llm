# LLM Project

## Project Task
The task is Sentiment Analysis on IMDB Movie Reviews to classify reviews as positive or negative.

## Dataset
The IMDB Movie Reviews dataset with 50,000 reviews, split into 25,000 for training and 25,000 for testing, was used.

## Pre-trained Model
I selected [BERT (Bidirectional Encoder Representations from Transformers)](https://github.com/google-research/bert) for its strong performance in NLP tasks, versatility for fine-tuning, and extensive community support. BERT demonstrated effective performance in initial experiments with our dataset.

## Performance Metrics
The model was evaluated using accuracy, precision, recall, and F1-score.

## Hyperparameters
Optimized hyperparameters included:
- **Learning Rate:** 3e-5
- **Batch Size (Training):** 32
- **Batch Size (Evaluation):** 64
- **Epochs:** 4
- **Weight Decay:** 0.01
- **Warmup Steps:** 500

## Model Training and Evaluation
The model was fine-tuned using the Hugging Face `Trainer` API. The process involved preprocessing the dataset, defining custom metrics, and training with the specified hyperparameters. The model achieved high accuracy and balanced precision and recall.

## Issues and Next Steps
There were issues uploading the model to Hugging Face Hub due to size constraints. Future work includes further hyperparameter tuning, exploring different models, and implementing real-time sentiment analysis.

## Model Output
The fine-tuned BERT model outputs labels ('POSITIVE' or 'NEGATIVE') with confidence scores. For example:
- **Text:** "I love this movie, it was fantastic!"
- **Prediction:** {'label': 'POSITIVE', 'score': 0.9998}

By fine-tuning BERT, the model effectively classifies movie reviews accurately.