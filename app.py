import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size: 48px;
            font-weight: 700;
            color: #4A90E2;
            text-align: center;
            padding: 10px;
        }
        .dev-name {
            text-align: center;
            font-size: 20px;
            font-weight: 500;
            color: #6c6c6c;
            margin-bottom: 30px;
        }
        hr {
            border: none;
            height: 2px;
            background-color: #f0f0f0;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .upload-section {
            margin-top: 20px;
            padding: 30px;
            background-color: #f5f5f5;  /* Softer background */
            border-radius: 8px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.08);
            border: 1px solid #e0e0e0; /* Subtle border */
        }
        .upload-section h3 {
            color: #333;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .upload-section label {
            font-size: 18px;
            color: #555;
            margin-bottom: 10px;
            display: block;
        }
        .upload-section .stTextArea, .upload-section .stFileUploader {
            width: 100%;
            max-width: 600px;
            margin-bottom: 20px;
        }
        .upload-section input {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# Beautiful header section
st.markdown('<div class="main-title">📄 Smart Resume Analyzer using NLP</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-name">Developed by <b>Agila Karunanithi</b></div>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to clean and preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Function to create a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Function to calculate similarity score
def get_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.header("Upload Resume and Job Description")

resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description Here")

if resume_file and jd_text:
    with st.spinner("Analyzing Resume..."):
        resume_text_raw = extract_text_from_pdf(resume_file)
        resume_text = preprocess(resume_text_raw)
        jd_cleaned = preprocess(jd_text)

        st.subheader("📊 Resume vs JD Similarity Score")
        similarity_score = get_similarity(resume_text, jd_cleaned)
        st.metric("Match Score", f"{similarity_score}%")

        st.subheader("☁️ Resume Word Cloud")
        generate_wordcloud(resume_text)

        st.subheader("📝 Suggestions")
        jd_tokens = set(jd_cleaned.split())
        resume_tokens = set(resume_text.split())
        missing_skills = jd_tokens - resume_tokens

        if missing_skills:
            st.write("Consider adding these relevant terms to your resume:")
            st.write(", ".join(list(missing_skills)[:15]))
        else:
            st.write("Your resume aligns well with the job description!")
else:
    st.info("Please upload a resume and paste the job description to begin analysis.")

st.markdown('</div>', unsafe_allow_html=True)
