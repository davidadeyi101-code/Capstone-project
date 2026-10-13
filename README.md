# Sentiment Analysis System

## Overview

This project is a Machine Learning-based Sentiment Analysis System developed as part of an AI/ML Capstone Project.

The system analyzes textual reviews on movies and predicts whether the sentiment expressed is **Positive** or **Negative**. By leveraging Natural Language Processing (NLP) techniques and machine learning algorithms, the project demonstrates how public opinions can be automatically classified, reducing the need for manual review analysis.

---

## Problem Statement

Movie industries receive large volumes of public feedback through reviews, comments, and surveys. Manually analyzing this information is time-consuming and inefficient.

This project addresses that challenge by developing an automated sentiment classification system capable of predicting the emotional tone of a review.

---

## Objectives

* Collect and prepare sentiment-labeled text data.
* Perform text preprocessing and feature extraction.
* Train a machine learning model for sentiment classification.
* Evaluate model performance using standard metrics.
* Provide an interface for testing custom reviews.

---

## Dataset

The project uses a sentiment-labeled movie review dataset obtained from Kaggle.

Dataset Features:

* Review Text
* Sentiment Label (Positive / Negative)

Dataset Source:

[https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews]

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* NLTK 

### Development Environment

* Visual Studio Code
* Git
* GitHub

### Deployment

* Streamlit

---

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas and inspected for quality and completeness.

### 2. Data Preprocessing

The following preprocessing steps were performed:

* Conversion to lowercase
* Removal of punctuation
* Removal of unnecessary characters
* Handling of missing values

### 3. Feature Extraction

Text reviews are converted into numerical representations using:

* CountVectorizer

### 4. Train-Test Split

The dataset is divided into:

* Training Set (80%)
* Testing Set (20%)


### 5. Model Training

A machine learning classification model is trained using the processed text features.

Model Selected:

[Logistic Regression]

Test models:

* Support Vector Machine
* Logistic Regression
* Random Forest

### 6. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
---

## Results

Model Performance:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 88% |
| Precision | 87% |
| Recall    | 89% |
| F1-Score  | 88% |

---

## Future Improvements

* Add Neutral sentiment classification.
* Experiment with TF-IDF vectorization.
* Compare multiple machine learning algorithms.
* Implement deep learning approaches such as LSTMs and Transformers.
* Deploy a fully interactive web application.

---

## Team Members

* [RAJI HAAMID ADEBAYO]
* [ADEYI OLUWAFERANMI DAVID]
* [OJO BUKOLA DAMILOLA]
* [HOGAN-ITAM ELIJAH]
* [LONGDUUT NAANSHUUT FARCIA]
* [AHMED SABDAT ONYECHE]
* [OGUNADE IBUKUNOLUWA SOLOMON]
---

## License

This project is licensed under the MIT License.

---

## Live Demo

Streamlit Application:

[]
