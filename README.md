# SMS Spam Classifier

## Overview
The **SMS Spam Classifier** is a machine learning project designed to accurately classify SMS messages as either **spam** or **ham** (non-spam). With the proliferation of spam messages, effective filtering is crucial for maintaining a positive user experience in messaging applications. This project employs Natural Language Processing (NLP) techniques and various machine learning algorithms to achieve high accuracy in spam detection.

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
The following technologies and libraries were used to develop this project:

- **Programming Language**: Python
- **Libraries**: 
  - **Pandas**: For data manipulation and analysis.
  - **Numpy**: For numerical computations and array handling.
  - **Scikit-learn**: For implementing machine learning algorithms and model evaluation.
  - **NLTK (Natural Language Toolkit)**: For text processing and natural language processing tasks.
  - **Flask**: For deploying the application as a web service.
- **Environment**: Jupyter Notebook was used for prototyping and experimentation.

## Dataset
The dataset used in this project is the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository. It consists of a collection of SMS messages that are labeled as "spam" or "ham." The dataset includes:

- **Messages**: Text of the SMS messages.
- **Labels**: A binary classification where 'spam' indicates a spam message and 'ham' indicates a legitimate message.

The dataset contains 5,574 messages, with 747 messages labeled as spam and 4,827 messages labeled as ham.

## Features
The SMS Spam Classifier project includes several key features:

- **Data Preprocessing**:
  - **Tokenization**: Breaking down the text into individual words or tokens.
  - **Stemming**: Reducing words to their base or root form to unify similar words (e.g., "running" to "run").
  - **Stop-word Removal**: Eliminating common words that do not contribute significant meaning to the classification (e.g., "and," "the").

- **Feature Extraction**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. This helps to emphasize important words while reducing the weight of common words.
  - **Bag-of-Words**: A simple representation of text data where the frequency of words is used as features for classification.

- **Machine Learning Models**:
  - Implemented multiple algorithms, including:
    - **Logistic Regression**: A statistical method for binary classification.
    - **Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.
    - **Support Vector Machines (SVM)**: A powerful classifier that works by finding the hyperplane that best separates the classes in the feature space.

- **Model Evaluation**:
  - Assessed model performance using metrics such as:
    - **Accuracy**: The overall correctness of the model.
    - **Precision**: The ratio of true positive results to the total predicted positives.
    - **Recall**: The ratio of true positive results to the actual positives.
    - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
