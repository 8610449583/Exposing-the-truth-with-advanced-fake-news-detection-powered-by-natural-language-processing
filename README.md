# 📰 Fake News Detection using NLP & Machine Learning

This project focuses on building a **fake news detection system** using Natural Language Processing (NLP) and Machine Learning techniques to automatically classify news articles as *real* or *fake*. It aims to combat misinformation spread across digital platforms and social media.

## 📌 Problem Statement

The rapid spread of misinformation poses serious threats to society, democracy, and trust in media. Manual fact-checking is slow and unsustainable. This project proposes an automated system for **binary classification of news articles** using textual features and machine learning models.

---

## 🎯 Objectives

- Detect fake news using NLP techniques.
- Preprocess and analyze real-world news datasets.
- Engineer relevant features from text and metadata.
- Train multiple classification models and evaluate their performance.
- Visualize model insights to support public awareness.

---

## 🧠 Type of Problem

- Binary Classification (Fake vs Real)
- Supervised Machine Learning
- Natural Language Processing (Text Classification)
- Social Media Analytics / Information Integrity

---

## 📂 Dataset

**Type**:  
- Unstructured Text: Title, content, etc.  
- Structured Metadata (optional): Author, date, topic

**Target Variable**: `Label`  
- Fake = 0, Real = 1  

**Sources**:  
- Public datasets from Kaggle, LIAR, FakeNewsNet, etc.

---

## 🔧 Data Preprocessing

- Handle missing values
- Remove duplicates
- Normalize text (lowercase, punctuation removal, etc.)
- Encode categorical features (author, subject)
- Convert data types (e.g., string to datetime)

---

## 📊 Exploratory Data Analysis (EDA)

- Univariate analysis: distribution of labels, article length
- Bivariate analysis: article length vs label
- Multivariate analysis: topic distribution, TF-IDF clustering, correlation matrix

---

## 🏗️ Feature Engineering

**Text Features:**
- Word count, sentence count, punctuation count
- Stopword ratio, sentiment scores (TextBlob/VADER)
- TF-IDF vectors, word embeddings (Word2Vec, BERT)

**Metadata Features (if available):**
- Encoded author/topic
- Publishing date features (day, month, etc.)

---

## 🤖 Model Building

- Baseline Models: Logistic Regression, Naive Bayes
- Advanced Models: Random Forest, SVM, XGBoost
- Deep Learning: LSTM, GRU, CNN
- Transformers: BERT, RoBERTa (using HuggingFace)

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix, ROC Curve, Precision-Recall Curve

---

## 📈 Visualization & Insights

- Confusion matrix for performance visualization
- ROC and PR curves for threshold analysis
- Feature importance for model interpretability


## 🛠️ Tools and Technologies

- **Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, seaborn, matplotlib, xgboost, plotly
- **NLP Tools**: NLTK, spaCy, HuggingFace Transformers
- **Development**: Jupyter Notebook, Google Colab
- **Version Control**: Git & GitHub

---

## 👥 Team Members & Contributions

- **M Sanjai Pravin** – Data Cleaning, Preprocessing, Feature Encoding  
- **V Sanjay Kumar** – EDA, Statistical Insights, Visualization  
- **K Sathya Sri** – Feature Engineering, Model Development  
- **S Sathya Priya** – Hyperparameter Tuning, Evaluation, Reporting

---

## 🔗 GitHub Repository

(https://github.com/8610449583/Exposing-the-truth-with-advanced-fake-news-detection-powered-by-natural-language-processing)

---

## College Name
Sri Ramanujar Engineering College – CSE Department  
