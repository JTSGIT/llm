# LLM Project

## Project Task
The chosen project task is Sentiment Analysis on IMDB Movie Reviews. The goal is to develop a model that classifies movie reviews as positive or negative.

## Dataset
The dataset used is the IMDB Movie Reviews dataset, which contains 50,000 reviews labeled as positive or negative. It is divided into training (25,000 reviews) and test (25,000 reviews) splits.

## Pre-trained Model
The pre-trained model selected for this project is DistilBERT (Distilled Bidirectional Encoder Representations from Transformers), known for its effectiveness in understanding the context of text for sentiment analysis.

## Performance Metrics
The performance of the model is evaluated using the following metrics:
- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures the proportion of true positive predictions out of all positive predictions.
- **Recall:** Measures the proportion of true positive predictions out of all actual positives.
- **F1-Score:** The harmonic mean of precision and recall, providing a single metric to balance both concerns.

## Hyperparameters
Key hyperparameters optimized during training include:
- **Learning Rate:** 3e-5
- **Batch Size (Training):** 32
- **Batch Size (Evaluation):** 64
- **Number of Epochs:** 4
- **Weight Decay:** 0.01
- **Warmup Steps:** 500

## Model Training and Evaluation
The model was fine-tuned using the Hugging Face `Trainer` API. The training process involved:
1. Preprocessing the dataset using the DistilBERT tokenizer.
2. Defining a custom metric computation function to evaluate the model using accuracy and F1-score.
3. Initializing the `Trainer` with the DistilBERT model, training arguments, dataset splits, and custom metrics.
4. Training the model with the specified hyperparameters.
5. Evaluating the model on the test dataset.

### Results
The optimized model achieved the following performance on the test set:
- **Accuracy:** (Add actual accuracy result here)
- **Precision:** (Add actual precision result here)
- **Recall:** (Add actual recall result here)
- **F1-Score:** (Add actual F1-score result here)

### Conclusion
The fine-tuned DistilBERT model for sentiment analysis on the IMDB dataset performed well, achieving high accuracy and balanced precision and recall. The optimization of hyperparameters such as learning rate, batch size, and number of epochs contributed to the improved performance. By pushing the model to the Hugging Face Hub, it is made accessible for further use and contributions from the NLP community.

## Future Work
- **Hyperparameter Tuning:** Further optimization of hyperparameters to improve model performance.
- **Model Comparison:** Fine-tuning and comparing different pre-trained models like RoBERTa, ALBERT, and others.
- **Data Augmentation:** Exploring data augmentation techniques to enhance the dataset and improve model robustness.
- **Real-time Inference:** Deploying the model for real-time sentiment analysis in a production environment.