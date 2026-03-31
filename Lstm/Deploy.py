import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Settings
MAX_WORDS = 15000
MAX_LEN = 150

nltk.download('stopwords')

# ============================
# Load Saved Model + Objects
# ============================
loaded_model = load_model("Final_Model/multi_model.h5")
loaded_tokenizer = joblib.load("Final_Model/tokenizer.pkl")
loaded_label_encoder_sentiment = joblib.load("Final_Model/label_encoder_sentiment.pkl")
loaded_label_encoder_capital = joblib.load("Final_Model/label_encoder_capital.pkl")

# ============================
# Preprocessing Functions
# ============================

def remove_unnecessary_characters(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

def tokenize_text(text):
    try:
        tokens = word_tokenize(str(text))
        return tokens
    except:
        return []

def normalize_text(tokens):
    cleaned = []
    for word in tokens:
        word = word.lower()
        word = re.sub(r'[^\w\s]', '', word)
        word = re.sub(r'\s+', ' ', word).strip()
        cleaned.append(word)
    return cleaned



def preprocess_input(text):
    # Step 1
    clean_text = remove_unnecessary_characters(text)

    # Step 2
    tokens = tokenize_text(clean_text)

    # Step 3
    norm_tokens = normalize_text(tokens)

    # Step 4
    stop_words = set(stopwords.words('english'))
    meaningful_words = {
        'not', 'no', 'nor', 'never', 'very', 'only',
        'few', 'many', 'most', 'much', 'against'
    }
    stop_words = stop_words - meaningful_words

    # Step 5
    filtered_words = [w for w in norm_tokens if w not in stop_words]
    return " ".join(filtered_words)

# ============================
# Prediction Loop
# ============================

print("\n🔁 Enter text for predictions. Type 'exit' to quit.\n")

while True:

    text = input("Enter the Text: ").strip()

    if text.lower() in ["exit", "quit", "q"]:
        print("\n✅ Exiting... Goodbye!\n")
        break

    preprocessed_text = preprocess_input(text)

    # Convert to sequence
    seq = loaded_tokenizer.texts_to_sequences([preprocessed_text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    # Predict
    sentiment_pred, capital_pred = loaded_model.predict(pad)

    sentiment_class = np.argmax(sentiment_pred, axis=1)
    capital_class = np.argmax(capital_pred, axis=1)

    decoded_sentiment = loaded_label_encoder_sentiment.inverse_transform(sentiment_class)[0]
    decoded_capital = loaded_label_encoder_capital.inverse_transform(capital_class)[0]

    # Output results
    print("\n--- Prediction Result ---")
    print("Input:", text)
    print("Cleaned:", preprocessed_text)
    print("Predicted Sentiment:", decoded_sentiment)
    print("Predicted Capital:", decoded_capital)
    print("\n---------------------------------------\n")
