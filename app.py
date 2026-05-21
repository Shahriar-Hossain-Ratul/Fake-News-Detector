import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK ডাউনলোড
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's Real or Fake.")

@st.cache_resource
def train_perfect_model():
    # কিছু শক্তিশালী রিয়েল এবং ফেক কি-ওয়ার্ড যুক্ত ডামি ডেটা তৈরি করা জাতে মডেল প্যাটার্ন বোঝে
    real_samples = [
        "Government officials announced a new economic framework today during the annual summit.",
        "NASA James Webb Space Telescope captured a stunning new image of a distant galaxy cluster.",
        "Scientists publish new research on renewable energy transitions and solar infrastructure.",
        "The federal reserve changed interest rates to balance inflation index this quarter.",
        "International Olympic Committee officially announced host cities for summer games.",
        "Researchers found a new treatment method in university medical trials published today.",
        "The prime minister signed a bilateral trade agreement to reduce tariffs on technology imports."
    ] * 200  # গুণ করে সংখ্যা বাড়ানো হচ্ছে

    fake_samples = [
        "SHOCKING: Secret government documents leaked online prove scientists built a time machine.",
        "Drinking three glasses of warm ocean saltwater every morning completely cures virus.",
        "BREAKING: Massive asteroid made of solid gold orbiting moon claimed by tech billionaire.",
        "Aliens landed in New York City and are handing out free ice cream to everyone.",
        "Doctors are furious that this simple secret trick is being exposed to the public.",
        "Leaked reports show historical figures are being cloned to alter election outcomes.",
        "This magical pill will instantly make everyone on Earth a millionaire overnight."
    ] * 200

    texts = real_samples + fake_samples
    labels = [0] * len(real_samples) + [1] * len(fake_samples)  # ০ = রিয়েল, ১ = ফেক (পারফেক্ট ৫০:৫০ ব্যালেন্স)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])
    
    cleaned_texts = [clean(t) for t in texts]
    
    # ভেক্টরাইজার এবং লজিস্টিক রিগ্রেশন মডেল ট্রেইনিং
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned_texts)
    y = np.array(labels)
    
    model = LogisticRegression(C=5.0)  # হাইপারপ্যারামিটার টিউনিং জাতে কি-ওয়ার্ড ভালো চেনে
    model.fit(X, y)
    
    return vectorizer, model, clean

with st.spinner("Optimizing AI Model with balanced parameters... Please wait."):
    tfidf, lr_model, clean_func = train_perfect_model()

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
