import streamlit as st
import pickle
import re
import time
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
st.set_page_config(
    page_title="TruthLens ML | Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = pickle.load(open("models/fake_news_model_decision_tree.pkl","rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl","rb"))



# ---------------- PREPROCESSING ----------------
# MATCHES YOUR NOTEBOOK clean_text()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- CSS DESIGN ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1e 100%);
    color:white;
}

/* Fix invisible headings */

h1,h2,h3,h4,h5,h6 {
color:white !important;
}

p, span, div {
color:#e6edf3;
}

/* Hero Section */

.hero {
text-align:center;
padding:60px 20px 40px 20px;
}

.hero-badge {
display:inline-block;
background:linear-gradient(90deg,#00d4ff22,#7b2ff722);
border:1px solid #00d4ff44;
color:#00d4ff;
padding:6px 20px;
border-radius:50px;
font-size:13px;
font-weight:600;
letter-spacing:2px;
margin-bottom:20px;
}

.hero-title {
font-size:72px;
font-weight:900;
background:linear-gradient(135deg,#ffffff,#00d4ff,#7b2ff7);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.hero-subtitle {
font-size:20px;
color:#9aa4b2;
max-width:650px;
margin:auto;
}

/* Stats */

.stats-row{
display:flex;
justify-content:center;
gap:20px;
margin-top:30px;
}

.stat-card{
background:linear-gradient(135deg,#ffffff08,#ffffff03);
border:1px solid #ffffff15;
border-radius:16px;
padding:20px 30px;
text-align:center;
}

.stat-number{
font-size:30px;
font-weight:800;
background:linear-gradient(135deg,#00d4ff,#7b2ff7);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.stat-label{
font-size:12px;
color:#8892a4;
}

/* Text area */

.stTextArea textarea{
background:#0d1117 !important;
border:1px solid #ffffff30 !important;
border-radius:14px !important;
color:white !important;
font-size:16px !important;
padding:20px !important;
}

/* Button */

.stButton > button{
background:linear-gradient(135deg,#00d4ff,#7b2ff7) !important;
color:white !important;
font-weight:700 !important;
border-radius:10px !important;
padding:14px 35px !important;
border:none !important;
}

/* Result cards */

.result-real{
background:linear-gradient(135deg,#00ff8815,#00ff8808);
border:1px solid #00ff8844;
border-radius:20px;
padding:30px;
text-align:center;
}

.result-fake{
background:linear-gradient(135deg,#ff000015,#ff000008);
border:1px solid #ff000044;
border-radius:20px;
padding:30px;
text-align:center;
}

/* Footer */

.footer{
text-align:center;
padding:40px;
color:#8892a4;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">

<div class="hero-badge">
Machine Learning Powered • 99.7% Accuracy
</div>

<div class="hero-title">
TruthLens ML
</div>

<div class="hero-subtitle">
Fake News Detection using NLP and Machine Learning with TF-IDF and Decision Tree classifier
</div>

</div>
""", unsafe_allow_html=True)

# ---------------- STATS ----------------

st.markdown("""
<div class="stats-row">

<div class="stat-card">
<div class="stat-number">99.7%</div>
<div class="stat-label">Accuracy</div>
</div>

<div class="stat-card">
<div class="stat-number">44K+</div>
<div class="stat-label">Articles Trained</div>
</div>

<div class="stat-card">
<div class="stat-number">3</div>
<div class="stat-label">Models Tested</div>
</div>

<div class="stat-card">
<div class="stat-number">&lt;1s</div>
<div class="stat-label">Detection Time</div>
</div>

</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- MAIN DETECTOR ----------------

col1,col2 = st.columns([1.2,1])

with col1:

    st.markdown("### Paste News Article")

    # Example news for demo
    example_news = "Scientists have discovered a new vaccine that could prevent future pandemics."

    if "news_text" not in st.session_state:
        st.session_state.news_text = ""

    # Demo button
    if st.button("Use Example News"):
        st.session_state.news_text = example_news

    news_text = st.text_area(
        "",
        height=280,
        value=st.session_state.news_text,
        placeholder="Paste any news article here..."
    )

    analyze_btn = st.button("Analyze Article")

with col2:

    st.markdown("### Prediction Result")

    if analyze_btn:

        if news_text.strip()=="":
            st.warning("Please enter a news article")

        else:

            with st.spinner("Analyzing article..."):

                time.sleep(1)

                processed = preprocess(news_text)

                vectorized = vectorizer.transform([processed])

                prediction = model.predict(vectorized)[0]

                probabilities = model.predict_proba(vectorized)[0]

                fake_prob = round(probabilities[0]*100,2)
                real_prob = round(probabilities[1]*100,2)

                confidence = max(fake_prob, real_prob)

            if prediction == 0:

             st.markdown("""
             <div class="result-fake">
             <h2>🚨 Fake News</h2>
             <p>This article shows strong indicators of misinformation.</p>
             </div>
             """, unsafe_allow_html=True)

             st.progress(fake_prob/100)
             st.write(f"Confidence: **{fake_prob}%**")

            else:

             st.markdown("""
             <div class="result-real">
             <h2>✅ Real News</h2>
             <p>This article appears legitimate.</p>
             </div>
             """, unsafe_allow_html=True)

            st.progress(real_prob/100)
            st.write(f"Confidence: **{real_prob}%**")

# ---------------- HOW IT WORKS ----------------

st.write("")
st.write("## How It Works")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.write("**1️⃣ Paste Article**")
    st.write("User inputs a news article")

with c2:
    st.write("**2️⃣ Text Preprocessing**")
    st.write("Lowercase + punctuation removal + stopword removal")

with c3:
    st.write("**3️⃣ TF-IDF Vectorization**")
    st.write("Text converted into numerical vectors")

with c4:
    st.write("**4️⃣ Decision Tree Prediction**")
    st.write("Model classifies news as Real or Fake")

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">

Built with Python • Scikit-learn • NLTK • Streamlit  

Models Tested: Naive Bayes • Logistic Regression • Decision Tree  

Final Model Used: Decision Tree Classifier

</div>
""", unsafe_allow_html=True)