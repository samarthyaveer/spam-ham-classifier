Spam/Ham Email Classifier
This project provides a machine learning-based solution for classifying emails as Spam or Ham. It leverages multiple models, including Naive Bayes, Support Vector Machines (SVM), and Random Forest, to predict and display results interactively via a Gradio interface. The application is deployed on Hugging Face for seamless accessibility.

Features
Multi-Model Predictions: Utilizes Naive Bayes, SVM, and Random Forest classifiers for predictions.
Interactive Interface: Accepts single email text or batch input through CSV files.
Preprocessing: Includes text preprocessing (stopword removal, lemmatization) and TF-IDF vectorization.
Deployment: Deployed on Hugging Face with a user-friendly interface for classification tasks.
Getting Started
Follow the instructions below to set up the project locally or use the deployed version on Hugging Face.

Usage
Single Email Classification
Enter email content in the Text Input field.
Click Submit to view the classification result for each model.
Batch Email Classification
Upload a CSV file containing a column with email texts.
Ensure the column name is recognized or renamed automatically.
Click Submit to receive predictions for all emails, including a majority vote result.
Deployed Version
Access the live demo on Hugging Face:
[https://hf.co/spaces/samarthyaveer/spam-ham-classifier]

How It Works
Preprocessing Pipeline
Convert text to lowercase.
Remove special characters and digits.
Tokenize and filter stopwords using NLTK.
Lemmatize tokens to reduce words to their base form.
Transform preprocessed text using TF-IDF vectorization.
Models
Naive Bayes: Probabilistic model based on word frequency.
Support Vector Machines (SVM): Linear classifier optimized for high-dimensional spaces.
Random Forest: Ensemble model aggregating decisions from multiple decision trees.
Output
Displays predictions from all three models.
Provides a final classification based on majority voting across the models.
Acknowledgments
Libraries Used: NLTK, Scikit-learn, Pandas, NumPy, Gradio
