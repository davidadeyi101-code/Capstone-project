import streamlit as st
import joblib

model = joblib.load("notebook/sentiment_model.pkl")
vectorizer = joblib.load("notebook/vectorizer.pkl")

st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬", layout="wide")
st.title("🎬 IMDb Sentiment Analyzer")

st.caption(
    "Enter a movie review and the model will predict whether "
    "the sentiment is Positive or Negative."
)
st.sidebar.header("About this model")
st.sidebar.metric("Model Accuracy", "88.0%")
st.sidebar.write(
    f"""
   - **Inputs:** Movie review by the user
   - **Outputs:** Sentiment prediction (Positive/Negative)
   """
)
st.sidebar.warning(
    "This model is trained on a dataset of movie reviews and may not be accurate for all types of reviews."
)

review = st.text_area("Movie Review")

if st.button("Analyze Sentiment" ):

    if review.strip():

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

    else:
        st.warning("Please enter a review.")