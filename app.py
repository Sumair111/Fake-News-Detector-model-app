# app.py
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

# Load once using cache
model = load_model()
vectorizer = load_vectorizer()

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Check if news you heard is Fake or Real using an Intelligent ML Model.")
st.write("Enter a news headline or article below:")

user_input = st.text_area("News Text", "")

if st.button("Predict"):
    if user_input.strip():
        cleaned_input = [user_input.lower().strip()]
        vectorized = vectorizer.transform(cleaned_input)
        prediction = model.predict(vectorized)
        label = "✅ Real News" if prediction[0] == 1 else "🚫 Fake News"
        st.success(f"Prediction: {label}")

        proba = model.predict_proba(vectorized)[0]
        confidence = max(proba) * 100
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")
