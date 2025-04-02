import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Simulated dataset: Expanded HC3-style (Human vs AI text)
data = {
    "text": [
        "This is a human-written answer with natural variations.",
        "AI-generated responses often have structured formatting.",
        "Humans may include typos and diverse sentence structures.",
        "Language models generate responses based on learned data.",
        "Writing by humans often includes unique expressions and errors.",
        "Large language models predict words based on probabilities.",
        "People write with emotion, style, and creativity.",
        "AI-generated content is consistent but lacks true originality.",
        "Grammar mistakes are common in human writing.",
        "AI tends to use formal and repetitive sentence structures."
    ],
    "label": ["human", "AI", "human", "AI", "human", "AI", "human", "AI", "human", "AI"]
}
df = pd.DataFrame(data)

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

df["processed_text"] = df["text"].apply(preprocess)

# Feature extraction
def extract_features(text):
    words = word_tokenize(text)
    word_count = len(words)
    char_count = sum(len(word) for word in words)
    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
    readability = flesch_reading_ease(text)
    return pd.Series([word_count, char_count, lexical_diversity, readability])

df[["word_count", "char_count", "lexical_diversity", "readability"]] = df["text"].apply(extract_features)

# Train AI text detection model
vectorizer = TfidfVectorizer(ngram_range=(1,2))  # Use bigrams for better text capture
X = vectorizer.fit_transform(df["processed_text"])
y = df["label"].apply(lambda x: 1 if x == "AI" else 0)

# Ensure both classes exist in the dataset
if len(set(y)) < 2:
    raise ValueError("Dataset must contain both AI and Human samples.")

# Train-test split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Use RandomForest for better generalization
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# AI text detection function
def detect_ai_text(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return "AI-Generated" if prediction == 1 else "Human-Written"

# Test with a sample text
sample_text = "This response is generated based on deep learning models."
print(f"Detection Result: {detect_ai_text(sample_text)}")
