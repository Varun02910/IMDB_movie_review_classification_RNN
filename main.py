#STEP1: import all the liberary
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

#Load the pretrained model with the ReLU activation
model = load_model('simple_rnn_imdb.h5')

##Step 2: Helper Functions
#Function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

##Step 3 : Predcition function
def predict_sentiment(review):
    preprocessed_test = preprocess_text(review)

    prediction = model.prediction(preprocessed_test)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


import streamlit as st
##Steamlit app
st.title('IMDB Moview Review Sentimanet Analysis')
st.write('Enter a moview to Classify it as positive or negative')

user_input = st.text_area('Moview Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    #Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.write(f'Sentimanet: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please entre a moview review.')