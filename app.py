import streamlit as st
import joblib, re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

st.title("Movie Review Sentiment Analyser")
st.write("Type a movie review and find out if it's positive or negative.")

review = st.text_area("Your review:", height=150)

if st.button("Analyse"):
    if review.strip():
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        if pred == 1:
            st.success(f"Positive review  ({prob[1]*100:.1f}% confidence)")
        else:
            st.error(f"Negative review  ({prob[0]*100:.1f}% confidence)")