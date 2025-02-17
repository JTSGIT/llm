# LLM Project

## Project Task
The task is Sentiment Analysis on IMDB Movie Reviews to classify reviews as positive or negative.

## Dataset
The IMDB Movie Reviews dataset with 50,000 reviews, split into 25,000 for training and 25,000 for testing, was used.

## EDA
For the EDA, I leveraged Python libraries to analyze text lengths, word frequencies, and sentiment-specific keywords. I utilized histograms to understand the distribution of review lengths and word clouds to visualize the most common words in positive and negative reviews. This helped identify key words that frequently appear in each sentiment category, aiding in the refinement of text preprocessing and feature extraction for the sentiment analysis model.

## Pre-trained Model
I selected [BERT (Bidirectional Encoder Representations from Transformers)](https://github.com/google-research/bert) for its strong performance in NLP tasks, versatility for fine-tuning, and extensive community support. BERT demonstrated effective performance in initial experiments with our dataset.

## Model Training & Performance Metrics
The model was evaluated using accuracy, precision, recall, and F1-score. The model was fine-tuned using the Hugging Face `Trainer` API. Additionally, I conducted tests in the (5-pretrain.ipynb) notebook, which confirmed that the model accurately interprets positive and negative movie reviews.

## Issues and Next Steps
There were issues uploading the model to Hugging Face Hub due to size constraints. Future work includes further hyperparameter tuning, exploring different models, and implementing real-time sentiment analysis.

## Model Output
The fine-tuned BERT model outputs labels ('POSITIVE' or 'NEGATIVE') with confidence scores. For example:
- **Text:** "I love this movie, it was fantastic!"
- **Prediction:** {'label': 'POSITIVE', 'score': 0.9998}

By fine-tuning BERT, the model effectively classifies movie reviews accurately.
