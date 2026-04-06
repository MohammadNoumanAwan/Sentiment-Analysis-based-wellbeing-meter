# Well-Being Meter v2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Settings
# -------------------------
MAX_WORDS = 15000
MAX_LEN = 150

# LSTM paths
MODEL_DIR = "Lstm"
LSTM_MODEL_PATH = f"{MODEL_DIR}/multi_model.h5"
LSTM_TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer.pkl"
LSTM_LABEL_ENCODER_SENTIMENT_PATH = f"{MODEL_DIR}/label_encoder_sentiment.pkl"
LSTM_LABEL_ENCODER_CAPITAL_PATH = f"{MODEL_DIR}/label_encoder_capital.pkl"

# SVM / LR paths
SVM_DIR = "Svm"
SVM_SENTIMENT_PATH = f"{SVM_DIR}/sentiment_model.pkl"
SVM_CAPITAL_PATH = f"{SVM_DIR}/capital_model.pkl"
SVM_SENTIMENT_ENCODER_PATH = f"{SVM_DIR}/label_encoder_sentiment.pkl"
SVM_CAPITAL_ENCODER_PATH = f"{SVM_DIR}/label_encoder_capital.pkl"

# -------------------------
# NLTK download
# -------------------------
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# -------------------------
# Custom NegationAwarePreprocessor Class
# (Required for SVM sentiment model - saved inside the pipeline)
# -------------------------
class NegationAwarePreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for negation-aware text preprocessing.
    This class is needed because it's part of the sentiment_pipeline saved model.
    """
    def __init__(self):
        self.negation_words = {
            'not', 'no', 'never', "n't", 'none', 'nobody', 
            'nothing', 'nowhere', 'neither', 'nor'
        }
    
    def _mark_negation(self, text):
        """Replace words in negated context with 'NOT_' prefix"""
        words = text.split()
        in_negation = False
        result = []
        
        for word in words:
            stripped = word.strip('.,!?;:"()[]{} ')
            lower = stripped.lower()
            
            # Check if this is a negation word
            if lower in self.negation_words or word.lower().endswith("n't"):
                in_negation = True
                result.append(word)
                continue
            
            # Check if negation scope ends (punctuation)
            if any(punct in word for punct in '.!?;') and in_negation:
                in_negation = False
            
            # Apply NOT_ prefix if in negation scope
            if in_negation and stripped:
                if lower not in {'not', "n't", 'no'}:
                    result.append(f"NOT_{word}")
                else:
                    result.append(word)
            else:
                result.append(word)
                
        return " ".join(result)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, str):
            return [self._mark_negation(X)]
        return [self._mark_negation(text) for text in X]

# -------------------------
# Preprocessing Functions
# -------------------------
def remove_unnecessary_characters(text):
    """Remove HTML tags and special characters"""
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

def tokenize_text(text):
    """Tokenize text using NLTK"""
    try:
        return word_tokenize(str(text))
    except:
        return str(text).split()

def preprocess_for_lstm(text):
    """
    Preprocessing for LSTM model
    - Remove special characters
    - Tokenize
    - Lowercase
    - Remove stopwords (preserve meaningful words like 'not')
    """
    # Clean text
    clean_text = remove_unnecessary_characters(text)
    
    # Tokenize
    tokens = tokenize_text(clean_text)
    
    # Lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove stopwords but preserve negation words
    stop_words = set(stopwords.words('english'))
    meaningful_words = {
        'not', 'no', 'nor', 'never', 'very', 'only', 
        'few', 'many', 'most', 'much', 'against'
    }
    stop_words = stop_words - meaningful_words
    
    # Filter
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    
    return " ".join(filtered)

def preprocess_for_svm(text):
    """
    Minimal preprocessing for SVM/LR models
    - Only basic cleaning
    - Pipelines handle the rest internally
    """
    return remove_unnecessary_characters(text)

# -------------------------
# Load LSTM Artifacts
# -------------------------
@st.cache_resource
def load_lstm():
    """Load LSTM model and associated artifacts"""
    try:
        model = load_model(LSTM_MODEL_PATH)
        tokenizer = joblib.load(LSTM_TOKENIZER_PATH)
        le_sent = joblib.load(LSTM_LABEL_ENCODER_SENTIMENT_PATH)
        le_cap = joblib.load(LSTM_LABEL_ENCODER_CAPITAL_PATH)
        return model, tokenizer, le_sent, le_cap
    except Exception as e:
        st.error(f"Error loading LSTM model: {str(e)}")
        return None, None, None, None

# -------------------------
# Load SVM/LR Artifacts
# -------------------------
@st.cache_resource
def load_svm():
    """Load SVM/LR models and associated artifacts"""
    try:
        svm_sentiment = joblib.load(SVM_SENTIMENT_PATH)
        svm_capital = joblib.load(SVM_CAPITAL_PATH)
        le_sent = joblib.load(SVM_SENTIMENT_ENCODER_PATH)
        le_cap = joblib.load(SVM_CAPITAL_ENCODER_PATH)
        return svm_sentiment, svm_capital, le_sent, le_cap
    except Exception as e:
        st.error(f"Error loading SVM/LR models: {str(e)}")
        return None, None, None, None

# -------------------------
# Prediction - LSTM
# -------------------------
def predict_lstm(text, model, tokenizer, le_sent, le_cap):
    """
    Predict using LSTM model
    Returns sentiment and capital predictions with probabilities
    """
    # Preprocess
    preprocessed = preprocess_for_lstm(text)
    
    # Convert to sequence
    seq = tokenizer.texts_to_sequences([preprocessed])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    
    # Predict
    predictions = model.predict(padded, verbose=0)
    sentiment_pred, capital_pred = predictions
    
    # Get labels
    s_idx = int(np.argmax(sentiment_pred[0]))
    c_idx = int(np.argmax(capital_pred[0]))
    
    sentiment_label = le_sent.inverse_transform([s_idx])[0]
    capital_label = le_cap.inverse_transform([c_idx])[0]
    
    return {
        "input": text,
        "cleaned": preprocessed,
        "sentiment_label": sentiment_label,
        "capital_label": capital_label,
        "sentiment_proba": sentiment_pred[0],
        "capital_proba": capital_pred[0],
        "sentiment_confidence": float(np.max(sentiment_pred[0])),
        "capital_confidence": float(np.max(capital_pred[0]))
    }

# -------------------------
# Prediction - SVM/LR
# -------------------------
def predict_svm(text, svm_sent, svm_cap, le_sent, le_cap):
    """
    Predict using SVM/LR models
    Both models are pipelines that handle their own preprocessing
    """
    # Minimal preprocessing (both pipelines expect this)
    preprocessed = preprocess_for_svm(text)
    
    # Capital prediction (Pipeline: TF-IDF → SelectKBest → SVM)
    capital_pred_encoded = svm_cap.predict([preprocessed])
    capital_label = le_cap.inverse_transform(capital_pred_encoded)[0]
    
    # Get capital probabilities
    try:
        capital_proba = svm_cap.predict_proba([preprocessed])[0]
        capital_confidence = float(np.max(capital_proba))
    except:
        capital_proba = None
        capital_confidence = None
    
    # Sentiment prediction (Pipeline: NegationPreprocessor → TF-IDF → LogisticRegression)
    sentiment_pred_encoded = svm_sent.predict([preprocessed])
    sentiment_label = le_sent.inverse_transform(sentiment_pred_encoded)[0]
    
    # Get sentiment probabilities
    try:
        sentiment_proba = svm_sent.predict_proba([preprocessed])[0]
        sentiment_confidence = float(np.max(sentiment_proba))
    except:
        sentiment_proba = None
        sentiment_confidence = None
    
    return {
        "input": text,
        "cleaned": preprocessed,
        "sentiment_label": sentiment_label,
        "capital_label": capital_label,
        "sentiment_confidence": sentiment_confidence,
        "capital_confidence": capital_confidence,
        "sentiment_proba": sentiment_proba,
        "capital_proba": capital_proba
    }

# -------------------------
# STREAMLIT APP
# -------------------------
def main():
    st.set_page_config(page_title="Well-Being Meter", layout="centered")
    st.title("😊 Sentiment Analysis Based Well-Being Meter")
    st.write("Choose your model below and start predicting!")

    # -------------------------
    # MODEL SELECTION
    # -------------------------
    model_choice = st.selectbox(
        "Select Model",
        ["LSTM Model", "SVM / Logistic Regression Model"]
    )

    # Load models in advance
    if model_choice == "LSTM Model":
        model, tokenizer, le_sent, le_cap = load_lstm()
        if model is None:
            st.error("Failed to load LSTM model. Please check file paths.")
            return
    else:
        svm_sent, svm_cap, le_sent, le_cap = load_svm()
        if svm_sent is None or svm_cap is None:
            st.error("Failed to load SVM/LR models. Please check file paths.")
            return

    st.subheader("Enter Text")
    user_text = st.text_area("Write something...", height=120)

    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Predicting..."):
                if model_choice == "LSTM Model":
                    res = predict_lstm(user_text, model, tokenizer, le_sent, le_cap)
                else:
                    res = predict_svm(user_text, svm_sent, svm_cap, le_sent, le_cap)

            # Show predictions
            st.success("Prediction Complete!")
            st.write("**Input:**", res['input'])
            st.write("**Cleaned:**", res['cleaned'])
            st.write("**Predicted Sentiment:**", res['sentiment_label'])
            st.write("**Predicted Capital:**", res['capital_label'])

            # Show probabilities for both models
            try:
                if 'sentiment_proba' in res and res['sentiment_proba'] is not None:
                    # Sentiment probabilities
                    sent_df = pd.DataFrame({
                        "label": le_sent.classes_,
                        "probability": res['sentiment_proba']
                    })
                    st.markdown("### Sentiment Probabilities")
                    st.dataframe(sent_df)
                    st.bar_chart(sent_df.set_index("label"))
                
                if 'capital_proba' in res and res['capital_proba'] is not None:
                    # Capital probabilities
                    cap_df = pd.DataFrame({
                        "label": le_cap.classes_,
                        "probability": res['capital_proba']
                    })
                    st.markdown("### Capital Probabilities")
                    st.dataframe(cap_df)
                    st.bar_chart(cap_df.set_index("label"))
            except Exception as e:
                st.info(f"Could not display probabilities: {str(e)}")

if __name__ == "__main__":
    main()
    print("\nTo run this app, use the command: streamlit run app.py")
