# SMS_Spam_Classifier


# SMS Spam Classifier

## Overview
The SMS Spam Classifier is a machine learning project aimed at classifying SMS messages as either spam or non-spam. This project employs natural language processing (NLP) techniques and various machine learning algorithms to effectively filter out unwanted messages.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - Pandas
  - Numpy
  - Scikit-learn
  - NLTK
  - Flask
- **Environment**: Jupyter Notebook

## Dataset
The dataset used for this project is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository. It contains a set of SMS messages labeled as "spam" or "ham" (non-spam).

## Features
- Data preprocessing: Tokenization, stemming, and removal of stop words.
- Feature extraction using TF-IDF and Bag-of-Words.
- Implementation of various classification algorithms: 
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines (SVM)
- Performance metrics: Accuracy, precision, recall, F1-score.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
