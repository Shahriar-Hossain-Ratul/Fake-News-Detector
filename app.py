import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK ডাউনলোড
nltk.download('stopwords')
nltk.download('wordnet')

# অ্যাপের শিরোনাম ও ডিজাইন
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's Real or Fake.")

# ক্যাশ মেমোরি ব্যবহার করে ব্যাকএন্ডে ডেটা লোড ও মডেল ট্রেইন করা (যাতে প্রতি ক্লিকে লোড না হয়)
@st.cache_resource
def load_and_train():
    # সরাসরি কাগলের ডেটাসেট সোর্স লিংক থেকে ছোট একটি অংশ অনলাইনে রিড করা
    true_url = "https://raw.githubusercontent.com/clmentbisaillon/Fake-and-real-news-dataset/master/True.csv"
    fake_url = "https://raw.githubusercontent.com/clmentbisaillon/Fake-and-real-news-dataset/master/Fake.csv"
    
    # অ্যাপটি যেন দ্রুত লোড হয় তাই প্রথম ১০০০টি করে রো নেওয়া হচ্ছে
    true_df = pd.read_csv(true_url, nrows=1000)
    fake_df = pd.read_csv(fake_url, nrows=1000)
    
    true_df['label'] = 0
    fake_df['label'] = 1
    
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df['total_text'] = df['title'] + " " + df['text']
    
    # টেক্সট ক্লিনিং
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])
    
    df['cleaned'] = df['total_text'].apply(clean)
    
    # ভেক্টরাইজার ও মডেল ট্রেইনিং
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return vectorizer, model, clean

# ব্যাকএন্ড রান করানো
with st.spinner("Initializing AI Model... Please wait a moment."):
    tfidf, lr_model, clean_func = load_and_train()

# ইউজার ইনপুট বক্স
user_input = st.text_area("Paste News Text Here:", height=200, placeholder="Type or paste the news content here...")

if st.button("Check Authenticity", type="primary"):
    if user_input.strip() == "":
        st.warning("Please paste some text first!")
    else:
        # প্রেডিকশন প্রসেস
        cleaned_input = clean_func(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = lr_model.predict(vectorized_input)[0]
        
        # ফলাফল প্রদর্শন
        st.write("---")
        if prediction == 1:
            st.error("🚨 ALERT: This looks like FAKE NEWS!")
        else:
            st.success("✅ VERIFIED: This looks like REAL NEWS.")
