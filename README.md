"Project Overview"

This project is a Sentiment Analysis Based Well-Being Meter that predicts the emotional sentiment of a given text and also identifies the capital category using Machine Learning models.

The application is built using Python and Streamlit and allows users to input text and get predictions using two different models:

LSTM Deep Learning Model
SVM + Logistic Regression Model

The system processes the input text, cleans and tokenizes it, and then predicts the sentiment and capital category with confidence scores.


"Project Structure"

project-folder/
│
├── app.py
├── README.md
│
├── Lstm/
│   ├── multi_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder_sentiment.pkl
│   └── label_encoder_capital.pkl
│
├── Svm/
│   ├── sentiment_model.pkl
│   ├── capital_model.pkl
│   ├── label_encoder_sentiment.pkl
│   └── label_encoder_capital.pkl


"How the System Works"

1. User Input

The user enters text in the Streamlit interface.

2. Text Preprocessing

The system performs preprocessing:
Remove unnecessary characters
Tokenization
Lowercasing
Stopword removal

3. Model Selection

The user selects one of the following models:
LSTM Model
SVM / Logistic Regression Model

4. Output Prediction

The model predicts:
Sentiment label
Capital label
Probability scores


"How to Run the Project"

1 Install Dependencies
pip install streamlit tensorflow scikit-learn nltk pandas numpy joblib

2 Run the Application
First run the main Python file:  app.py
After running this command, the terminal will display a message like:
To run this app, use the command: streamlit run app.py
Now run the following command in the terminal:  streamlit run app.py

3 Open in Browser
After running the Streamlit command, the application will automatically open in your browser at:
http://localhost:8501

Example Usage
Select LSTM Model or SVM / Logistic Regression Model
Enter text such as:
I feel very happy today and everything is going great!
Click Predict

The system will display:
Predicted Sentiment
Predicted Capital
Probability charts for each prediction

