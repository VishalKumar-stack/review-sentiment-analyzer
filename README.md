# E-Commerce Review Sentiment Analyzer

## Project Overview

This project implements a Natural Language Processing (NLP) pipeline to classify e-commerce product reviews as positive or negative.

The project demonstrates how unstructured text can be converted into numerical features and used for supervised machine learning.

## Objective

The objective is to automatically classify customer reviews into:

- Positive
- Negative

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Regular Expressions

## NLP Workflow

1. Create a synthetic e-commerce review dataset
2. Clean and normalize review text
3. Split the data into training and testing sets
4. Convert text into numerical features using TF-IDF
5. Train a Logistic Regression classifier
6. Evaluate the model
7. Perform 5-fold cross-validation
8. Compare Logistic Regression with Naive Bayes
9. Test the model on new unseen reviews

## Text Preprocessing

The preprocessing includes:

- Converting text to lowercase
- Removing special characters and numbers
- Removing unnecessary whitespace

## Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert textual reviews into numerical feature vectors.

Both unigrams and bigrams were considered.

## Machine Learning Models

Two classical machine learning algorithms were evaluated:

### Logistic Regression

Used as the primary sentiment classification model.

### Multinomial Naive Bayes

Used as a second classical NLP baseline for comparison.

## Evaluation

The Logistic Regression model produced:

- Test Accuracy: 50.00%
- 5-Fold Cross-Validation Accuracy: 67.50%

The Naive Bayes model produced:

- 5-Fold Cross-Validation Accuracy: 67.50%

## Dataset

The dataset is synthetically generated for educational purposes.

It contains 80 labelled product reviews:

- 40 Positive
- 40 Negative

## Limitations

The dataset is relatively small and synthetic, so the evaluation results should not be interpreted as production-level sentiment performance.

The vocabulary and writing styles are also limited compared with real-world customer reviews.

## Future Improvements

Possible improvements include:

- Training on a larger real-world labelled review dataset
- Hyperparameter tuning
- More extensive text preprocessing
- Error analysis of misclassified reviews
- Handling negation and contextual sentiment
- Comparing with modern transformer-based NLP models

## Key Learning

This project demonstrates the complete basic NLP classification workflow from raw text preprocessing and TF-IDF feature extraction to supervised learning, cross-validation and model comparison.