# IMDB Movie Review Sentiment Analysis 🎬

## Overview
This project is an **IMDB Movie Review Sentiment Analysis** application built using **TensorFlow/Keras** and **Streamlit**. It uses a Simple Recurrent Neural Network (RNN) deep learning model to predict whether a given movie review has a positive or negative sentiment. 

The model was trained on the standard IMDB dataset and evaluates the input text to determine the likelihood of it being a positive review.

## Features
- **Deep Learning Model:** Utilizes a pre-trained Simple RNN model (`simple_Rnn.h5`).
- **Interactive UI:** A simple, clean, and intuitive web interface built with Streamlit.
- **Real-time Prediction:** Users can write or paste a movie review and instantly get a sentiment classification (Positive or Negative) along with a confidence score.
- **Automated Text Preprocessing:** Automatically tokenizes and pads the user input text to match the expected input shape (maxlen=500) for the model.

## Technologies Used
- Python
- TensorFlow / Keras (for building and running the RNN model)
- Streamlit (for the interactive frontend application)
- NumPy

## Repository Contents
- `main.py`: The Streamlit web application script.
- `simple_Rnn.h5`: The trained Recurrent Neural Network model file.
- `simpleRnn.ipynb` / `prediction.ipynb` / `embedding.ipynb`: Jupyter Notebooks utilized for data exploration, building the model, hyperparameter testing, and embedding analysis.
- `requirements.txt`: The list of Python dependencies required to run this project.
