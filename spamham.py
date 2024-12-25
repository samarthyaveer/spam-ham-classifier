import numpy as np
import pandas as pd
import nltk
import re
import gdown
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
import io

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Download dataset from Google Drive using gdown
url = 'https://drive.google.com/uc?id=1xu6FOMPERE0uflcAfjXDxV6W45IYOmT3'
gdown.download(url, 'spam_ham.csv', quiet=False)

# Step 2: Load the dataset
data = pd.read_csv('spam_ham.csv')

# Step 3: Identify and rename text column to 'EmailText'
text_column = None
for col in data.columns:
    if data[col].dtype == object:
        avg_text_len = data[col].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()
        if avg_text_len > 20:
            text_column = col
            break

if text_column:
    data.rename(columns={text_column: 'EmailText'}, inplace=True)
else:
    raise ValueError("No suitable text column found.")

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    tokens = text.split()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['ProcessedText'] = data['EmailText'].apply(preprocess_text)

# Vectorize the 'ProcessedText' using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['ProcessedText']).toarray()

# Target variable (assuming 'Label' column contains spam/ham labels)
y = data['spam']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
nb_model = MultinomialNB().fit(X_train, y_train)
svm_model = SVC(kernel='linear').fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
evaluate_model(svm_model, X_test, y_test, "SVM")
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Define classification function for single email
def classify_text_input(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    predictions = {
        "Naive Bayes": nb_model.predict(vectorized_text)[0],
        "SVM": svm_model.predict(vectorized_text)[0],
        "Random Forest": rf_model.predict(vectorized_text)[0],
    }
    print("Predictions for the input text:", predictions)

# Define classification function for multiple emails
def classify_csv_file(file):
    data = pd.read_csv(file)

    # Check and rename the text column to 'EmailText' if it's not present
    if 'EmailText' not in data.columns:
        potential_text_columns = {
            col: data[col].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()
            for col in data.columns
        }
        text_column = max(potential_text_columns, key=potential_text_columns.get)
        data.rename(columns={text_column: 'EmailText'}, inplace=True)

    # Proceed only if 'EmailText' column is correctly identified
    if 'EmailText' not in data.columns:
        print("Could not identify a suitable text column.")
        return

    # Preprocess and classify
    data['ProcessedText'] = data['EmailText'].apply(preprocess_text)
    vectorized_data = vectorizer.transform(data['ProcessedText']).toarray()
    nb_pred = nb_model.predict(vectorized_data)
    svm_pred = svm_model.predict(vectorized_data)
    rf_pred = rf_model.predict(vectorized_data)

    # Majority vote
    data['NaiveBayesPrediction'] = nb_pred
    data['SVMPrediction'] = svm_pred
    data['RandomForestPrediction'] = rf_pred
    data['FinalPrediction'] = data[['NaiveBayesPrediction', 'SVMPrediction', 'RandomForestPrediction']].mode(axis=1)[0]

    display(data[['EmailText', 'NaiveBayesPrediction', 'SVMPrediction', 'RandomForestPrediction', 'FinalPrediction']])

    # Display spam and ham counts
    spam_count = (data['FinalPrediction'] == 1).sum()  # Assuming 1 is 'spam'
    ham_count = (data['FinalPrediction'] == 0).sum()  # Assuming 0 is 'ham'
    total_mails = len(data)
    print(f"\nOut of {total_mails} emails:")
    print(f"Spam count: {spam_count}")
    print(f"Ham count: {ham_count}")

# UI Setup
option_selector = widgets.RadioButtons(
    options=['Single Email', 'Multiple Emails'],
    description='Choose Option:',
    disabled=False
)

text_input = widgets.Textarea(
    placeholder='Enter the email text to classify...',
    description='Email Text:',
    layout=widgets.Layout(width='100%', height='100px'),
    visible=False
)

classify_button = widgets.Button(description="Classify Email", visible=False)

upload_button = widgets.FileUpload(
    accept='.csv',
    multiple=False,
    description='Upload CSV',
    visible=False
)

# Display option selector
def on_option_change(change):
    clear_output()
    display(option_selector)
    if change['new'] == 'Single Email':
        # Show the text input and classify button for single email
        text_input.visible = True
        classify_button.visible = True
        upload_button.visible = False
        display(text_input, classify_button)
    elif change['new'] == 'Multiple Emails':
        # Show the upload button for multiple emails
        text_input.visible = False
        classify_button.visible = False
        upload_button.visible = True
        display(upload_button)

# Single email classification button action
def on_classify_single(b):
    classify_text_input(text_input.value)

# Multiple email file upload action
def on_file_upload(change):
    file = next(iter(upload_button.value.values()))['content']
    classify_csv_file(io.BytesIO(file))

# Bind actions
option_selector.observe(on_option_change, names='value')
classify_button.on_click(on_classify_single)
upload_button.observe(on_file_upload, names='value')

# Display initial option selector
display(option_selector)
