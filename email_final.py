import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Sample emails and their departments (labels)
emails = [
    ("Customer complaint: Issue with order", "Customer Service"),
    ("Technical support needed for server", "IT Support"),
    ("Billing inquiry about recent charges", "Finance"),
    ("Product feature request", "Product Development"),
    ("HR policy clarification", "Human Resources")
]

# Function to preprocess text (tokenization and lemmatization)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    tokens = text.split()  # Split text into words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Prepare data
X = [preprocess_text(email[0]) for email in emails]
y = [email[1] for email in emails]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and grid search parameters
pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB()
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'tfidfvectorizer__max_df': [0.75, 1.0],
    'tfidfvectorizer__min_df': [1, 2],
    'tfidfvectorizer__max_features': [None, 5000, 10000],
    'multinomialnb__alpha': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Function to predict department from email content
def predict_department(email_content):
    cleaned_content = preprocess_text(email_content)
    predicted_department = best_model.predict([cleaned_content])[0]
    return predicted_department

# Test the classifier
test_email = "IT hope this message finds you well. HR am writing to propose an exciting new feature enhancement for our platform that HR believe will significantly enhance user engagement and satisfaction.After conducting extensive user feedback sessions and analyzing market trends, we have identified a critical need for a more advanced and customizable user dashboard. The current dashboard, while functional, lacks the flexibility and personalization options that our users are increasingly demanding. invite your feedback and suggestions on the proposed feature set. Please feel free to share any additional ideas or considerations that you believe would contribute to the success of this initiative.Let's schedule a meeting early next week to discuss the feasibility and implementation timeline. Your insights are invaluable as we work towards delivering a superior user experience.Thank you for your attention to this matter. HR look forward to our discussion and moving this project forward."
predicted_department = predict_department(test_email)
print(f"The email '{test_email}' is likely from the department: {predicted_department}")
