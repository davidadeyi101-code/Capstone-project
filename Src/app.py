import streamlit as st
import joblib
import os

print("Current directory:", os.getcwd())
print("Notebook exists:", os.path.exists("notebook"))
print("Model exists:", os.path.exists("notebook/sentiment_model.pkl"))
print("Vectorizer exists:", os.path.exists("notebook/vectorizer.pkl"))

# Load model and vectorizer
model = joblib.load("notebook/sentiment_model.pkl")
vectorizer = joblib.load("notebook/vectorizer.pkl")

st.title("🎬 IMDb Sentiment Analyzer")

st.write(
    "Enter a movie review and the model will predict whether "
    "the sentiment is Positive or Negative."
)

review = st.text_area("Movie Review")

if st.button("Analyze Sentiment"):

    if review.strip():

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

    else:
        st.warning("Please enter a review.")