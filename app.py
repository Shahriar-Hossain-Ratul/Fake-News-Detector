import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK রিসোর্স ডাউনলোড
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's Real or Fake.")

@st.cache_resource
def load_and_train():
    # শাফেলড ডেটা রিড করা
    df = pd.read_csv('cleaned_news_sample.csv')
    
    if 'total_text' not in df.columns:
        df['total_text'] = df['title'] + " " + df['text']
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])
    
    df['cleaned'] = df['total_text'].apply(clean)
    
    # এন-গ্রাম (ngram_range) সহ ভেক্টরাইজার টিউনিং জাতে শব্দযুগল ভালো চেনে
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']
    
    # রেগুলারাইজেশন সহ লজিস্টিক রিগ্রেশন
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    
    return vectorizer, model, clean

with st.spinner("Initializing AI Model with balanced dataset... Please wait."):
    tfidf, lr_model, clean_func = load_and_train()

user_input = st.text_area("Paste News Text Here:", height=200, placeholder="Type or paste the news content here...")

if st.button("Check Authenticity", type="primary"):
    if user_input.strip() == "":
        st.warning("Please paste some text first!")
    else:
        cleaned_input = clean_func(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = lr_model.predict(vectorized_input)[0]
        
        st.write("---")
        if prediction == 1:
            st.error("🚨 ALERT: This looks like FAKE NEWS!")
        else:
            st.success("✅ VERIFIED: This looks like REAL NEWS.")
            st.success("✅ VERIFIED: This looks like REAL NEWS.")
            st.success("✅ VERIFIED: This looks like REAL NEWS.")
