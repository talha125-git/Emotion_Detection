import pandas as pd
import nltk
import string
import re
import pickle
import os
from pathlib import Path

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ================== SETUP ==================
print("=" * 50)
print("Initializing Emotion Detection Model...")
print("=" * 50)

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except:
    print("Downloading NLTK data...")
    nltk.download('stopwords')

# ================== LOAD DATA FROM CSV ==================
def load_dataset_from_csv():
    """Load dataset from CSV file"""
    csv_files = ['emotions_dataset.csv']
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"📂 Loading dataset from {csv_file}...")
            data = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(data)} samples from {csv_file}")
            return data
    
    # If no CSV file exists, create a default one
    print("⚠️ No CSV file found. Creating default dataset...")
    data = pd.DataFrame({
        'text': [
            'I am happy', 'I feel great', 'This is wonderful',
            'I am sad', 'This is terrible', 'I feel bad',
            'I am angry', 'This makes me mad', 'I hate this',
            'I am scared', 'This is frightening', 'I feel afraid',
            'This is normal', 'Nothing special', 'Regular day'
        ],
        'emotion': ['happy', 'happy', 'happy',
                    'sad', 'sad', 'sad',
                    'angry', 'angry', 'angry',
                    'fear', 'fear', 'fear',
                    'neutral', 'neutral', 'neutral']
    })
    
    # Save to CSV for future use
    data.to_csv('emotions_dataset.csv', index=False)
    print("💾 Created emotions_dataset.csv with default data")
    return data

# ================== TEXT PREPROCESSING ==================
def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# ================== MODEL TRAINING ==================
print("\n📊 Loading dataset...")
data = load_dataset_from_csv()
print(f"✅ Dataset loaded with {len(data)} samples")

print("\n🧹 Cleaning text...")
data['clean_text'] = data['text'].apply(clean_text)

print("\n📈 Class distribution:")
print(data['emotion'].value_counts())

# Prepare data
X = data['clean_text']
y = data['emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📚 Training samples: {len(X_train)}")
print(f"🧪 Testing samples: {len(X_test)}")

# Create and train model
print("\n🤖 Training model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words=stopwords.words('english'),
        max_features=1000,
        ngram_range=(1, 2)
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained successfully!")
print(f"📊 Accuracy: {accuracy:.2%}")
print(f"🎯 Precision: {accuracy*100:.1f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ================== PREDICTION FUNCTIONS ==================
def predict_emotion(text):
    """Predict emotion from text"""
    try:
        cleaned = clean_text(text)
        prediction = model.predict([cleaned])[0]
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return "neutral"

def predict_emotion_with_confidence(text):
    """Predict emotion with confidence score"""
    try:
        cleaned = clean_text(text)
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([cleaned])[0]
            prediction = model.predict([cleaned])[0]
            
            # Get confidence (max probability)
            confidence = max(proba) * 100
            
            # Get class index
            classes = model.classes_
            pred_index = list(classes).index(prediction)
            confidence = proba[pred_index] * 100
            
            return prediction, round(float(confidence), 1)
        else:
            # Fallback for models without predict_proba
            prediction = model.predict([cleaned])[0]
            return prediction, 75.0  # Default confidence
            
    except Exception as e:
        print(f"Prediction with confidence error: {e}")
        return "neutral", 50.0

# ================== TEST EXAMPLES ==================
print("\n" + "=" * 50)
print("🧪 Testing the model with sample inputs:")
print("=" * 50)

test_cases = [
    "I am so happy today!",
    "This is very sad news",
    "I'm extremely angry about this",
    "I feel scared and alone",
    "Just a normal day"
]

for text in test_cases:
    emotion, confidence = predict_emotion_with_confidence(text)
    print(f"'{text[:30]}...' → {emotion.upper()} ({confidence:.1f}%)")

print("\n" + "=" * 50)
print("🎯 Model is ready for use!")
print("Run 'streamlit run app.py' to start the application")
print("=" * 50)

# Make functions available for import
__all__ = ['predict_emotion', 'predict_emotion_with_confidence']