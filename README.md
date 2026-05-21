# Fake News Detector (Machine Learning Project)

An end-to-end Machine Learning pipeline to classify news articles as Real or Fake using NLP and Classification models.

## Dataset
The dataset used in this project is from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
*(Please download `True.csv` and `Fake.csv` from the link to run the notebook).*

## Project Pipeline Covered:
- **Data Acquisition & EDA:** Visualized class distribution using Seaborn.
- **Preprocessing:** Text cleaning, lowercasing, stopword removal, and Lemmatization using NLTK.
- **Feature Extraction:** Transformed text to numbers using `TfidfVectorizer`.
- **Model Selection:** Trained a baseline **Logistic Regression** and a complex **Random Forest Classifier**.
- **Hyperparameter Tuning:** Tuned Random Forest using `GridSearchCV`.
- **Rigorous Evaluation:** Evaluated using Classification Report (F1-Score) and Confusion Matrix.

## Results
- **Logistic Regression Accuracy:** 98%
- **Tuned Random Forest Accuracy:** 100%
