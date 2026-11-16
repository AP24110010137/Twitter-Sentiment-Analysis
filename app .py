
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_resources():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

best_model, tfidf = load_resources()

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not", "no", "nor", "never"}

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)

    if not cleaned:
        return None

    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return labels[prediction]

st.title("Twitter Sentiment Analysis")
st.subheader("Using Machine Learning")

st.write(
    "Enter any English text or tweet below. "
    "Our machine learning model will predict its sentiment."
)

st.divider()

user_text = st.text_area(
    "Enter your text here:",
    placeholder="Example: I really enjoyed this experience!",
    height=150
)

if st.button("Predict Sentiment", type="primary", use_container_width=True):

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        result = predict_sentiment(user_text)

        if result is None:
            st.warning("Please enter meaningful English text.")

        elif result == "Positive":
            st.success("Predicted Sentiment: POSITIVE 😊")

        elif result == "Negative":
            st.error("Predicted Sentiment: NEGATIVE")

        else:
            st.info("Predicted Sentiment: NEUTRAL")

st.divider()

st.caption(
    "Project: Sentiment Analysis on Social Media (Twitter) "
    "Using Machine Learning"
)

st.caption("Developed as an academic machine learning project.")
