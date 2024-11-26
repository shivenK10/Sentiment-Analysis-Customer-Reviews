Overview
This project performs sentiment analysis on a dataset of textual reviews, aiming to classify sentiments as positive or negative. It combines traditional machine learning models (Logistic Regression, SVM) with deep learning architectures (ANN, LSTM, GRU) and leverages pretrained GloVe embeddings to enhance the model's performance.

Features
Text Preprocessing: Includes steps like lowercasing, punctuation removal, number removal, tokenization, and lemmatization using NLTK's WordNet Lemmatizer.
Tokenization and Word Embeddings: Uses Keras's Tokenizer for numerical representation of text and GloVe embeddings for semantic context.
Multi-Model Approach:
Logistic Regression and SVM for baseline comparisons.
Deep learning models (ANN, LSTM, GRU) for advanced sentiment classification.
Interactive Prediction System: Allows users to input custom text and get real-time sentiment classification.
Model Comparison: Includes a bar chart comparing the accuracy of various models, helping visualize their performance.
Regularization: Implements dropout layers and early stopping to reduce overfitting.
Project Pipeline

Text Preprocessing:
Clean text by removing special characters, punctuation, and numbers.
Tokenize text and apply lemmatization to standardize words.

Feature Engineering:
Use TF-IDF vectorization for traditional models.
Integrate pretrained GloVe embeddings for deep learning models.
Model Training:

Train Logistic Regression and SVM models using TF-IDF features.
Train ANN, LSTM, and GRU models using padded sequences and GloVe embeddings.
Evaluation and Visualization:

Compare the performance of all models using accuracy scores.
Visualize results using a bar chart.
Interactive Prediction:

Build a system where users can input text and receive sentiment predictions in real time.
Technologies and Tools
Programming Language: Python
Libraries:
Text Preprocessing: NLTK, re, Keras
Machine Learning Models: scikit-learn
Deep Learning Models: TensorFlow, Keras
Visualization: Matplotlib
Word Embeddings: Pretrained GloVe embeddings (glove.6B.100d)
Installation
Prerequisites
Python 3.8 or later
Required libraries (install via pip):
pip install nltk tensorflow keras scikit-learn matplotlib
Download GloVe embeddings (glove.6B.100d.txt) from GloVe website.
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
Place the GloVe file in the appropriate directory (e.g., data/glove.6B.100d.txt).

The dataset used in this project contains:
Columns:
text: The review text.
label: The sentiment label (0 for negative, 1 for positive).

Size: 150,000 reviews.

Results

Performance Comparison:
Logistic Regression: ~89% accuracy
SVM: ~89% accuracy
ANN: ~88% accuracy
LSTM: ~90% accuracy
GRU: ~90% accuracy

Interactive Sentiment Prediction:
Predicts the sentiment of custom input text with high accuracy and confidence.

Applications
E-commerce: Analyze product reviews to identify customer satisfaction.
Social Media: Monitor brand sentiment or public opinion on events.
Customer Feedback: Extract insights from feedback to improve services.

Future Improvements
Use Transformer-based embeddings like BERT or GPT for enhanced performance.
Handle multi-class sentiment classification (e.g., positive, neutral, negative).
Build a web or mobile application for wider accessibility.

Acknowledgments
Stanford GloVe for pretrained word embeddings.
NLTK and Keras for NLP and deep learning support.
