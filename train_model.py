import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

print("Loading dataset...")
data = pd.read_csv('dataset/dataset.csv')
print("Dataset loaded successfully!")

print("\nSample of your data:")
print(data.head())

print("\nCreating model pipeline...")
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Converts text to numbers
    ('classifier', MultinomialNB())    # Simple classification algorithm
])

print("Training model...")
model.fit(data['text'], data['label'])

print("Saving model...")
joblib.dump(model, 'bad_word_model.joblib')
print("\nModel trained and saved as 'bad_word_model.joblib'!")
print("You can now use app.py to run the Flask server.")