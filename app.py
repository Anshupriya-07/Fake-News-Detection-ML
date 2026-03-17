import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# load model and vectorizer
model = pickle.load(open("models/fake_news_model_decision_tree.pkl","rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl","rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def predict_news(news):

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 0:
        return "🚨 Fake News"
    else:
        return "✅ Real News"


st.title("📰 Fake News Detection App")

st.write(
"This app uses Machine Learning and NLP to classify news articles as **Fake or Real**."
)

news = st.text_area("Paste News Article Here")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news text")

    else:
        result = predict_news(news)
        st.success(result)