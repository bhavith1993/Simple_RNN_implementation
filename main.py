# Import all the libraries

import numpy as np  
import tensorflow as tf     
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import sequence

# Load the word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model with RElu activation

model = load_model('simple_rnn_model.h5')

## Helper functions 

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Function to preprocess text

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function

## Prediction function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    
    prediction=model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]
    
    
import streamlit as st

# Streamlit app
st.title("IDMB movie review sentiment analysis")
st.write("Enter a movie review to predict its sentiment.")

# User input 

user_input = st.text_area("Enter your review here:")

if st.button("Classify"):
    
    preprocess_input = preprocess_text(user_input)
    
    # Make prediction
    
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    
    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]}")
    
else:
    st.write('Please enter a review')
    
    