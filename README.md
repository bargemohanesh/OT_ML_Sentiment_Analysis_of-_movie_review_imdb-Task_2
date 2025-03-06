# OT_ML_Sentiment_Analysis_of-_movie_review_imdb-Task_2
Sentiment Analysis of movie review (imdb)
Sentiment Analysis on Movie Reviews

Overview

This project performs sentiment analysis on IMDB movie reviews using machine learning techniques. The goal is to classify reviews as either positive or negative based on their textual content.

Dataset

Source: IMDB 50K Movie Reviews Dataset

Columns:

review: Textual content of the movie review

sentiment: Sentiment label (Positive/Negative)

Preprocessing Steps

Text Cleaning: Remove special characters, numbers, and punctuation.

Lowercasing: Convert all text to lowercase.

Tokenization: Split text into individual words.

Stopword Removal: Remove common words that do not contribute to sentiment.

Lemmatization/Stemming: Reduce words to their base form.

Feature Engineering

TF-IDF Vectorization: Convert text data into numerical format using Term Frequency-Inverse Document Frequency (TF-IDF).

Machine Learning Models

The following models are trained and evaluated:

Naïve Bayes (MultinomialNB)

Logistic Regression

Model Evaluation

Metrics Used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix Visualization

Installation & Requirements

Dependencies

Ensure you have the following libraries installed:

pip install pandas numpy scikit-learn nltk spacy tqdm matplotlib seaborn

Running the Project

Clone the repository:

git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis

Download necessary NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

Run the Jupyter Notebook or Python script:

jupyter notebook OT_ML_Sentiment_Analysis_On_Movie_Review.ipynb

Saving Preprocessed Data

To avoid preprocessing again after a kernel restart, save the cleaned dataset:

df.to_csv('preprocessed_reviews.csv', index=False)

Contributing

Feel free to fork this repository and submit pull requests for improvements!

License

This project is open-source and available under the MIT License.
