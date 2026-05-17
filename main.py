import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
word_index = {k: v for k, v in word_index.items() if v < 10000}
reversed_word_index = {value: key for key, value in word_index.items()}

model = load_model("simple_Rnn.h5")


def decode_review(encoded_review):
    return " ".join([
        reversed_word_index.get(i - 3, '?') for i in encoded_review
    ])


def preprocess_text(text):
    words = text.lower().split()

    encoded_review = []

    for word in words:
        index = word_index.get(word, 2)

        if index >= 10000:
            index = 2

        encoded_review.append(index + 3)

    return sequence.pad_sequences([encoded_review], maxlen=500)


st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and classify it as Positive or Negative")

user_input = st.text_area("Write your movie review here:")

if st.button("Button"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        processed_input = preprocess_text(user_input)

        prediction = model.predict(processed_input)

        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {prediction[0][0]:.4f}")

else:
    st.info("Type a review and click Classify")