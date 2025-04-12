import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import os

# Set page configuration
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

# Title of the app
st.title("📄 Smart Resume Analyzer using NLP")

# File uploader for the resume
uploaded_file = st.file_uploader("Upload Resume (PDF format only)", type=["pdf"])

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# If the user uploads a resume
if uploaded_file is not None:
    # Extract text from the PDF
    resume_text = extract_text_from_pdf(uploaded_file)

    # Analyze text using spaCy
    doc = nlp(resume_text)

    # Extracting keywords using TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text])

    # You can modify this part to analyze the resume further
    keywords = vectorizer.get_feature_names_out()
    
    # Wordcloud to visualize the most frequent words in the resume
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(resume_text)
    
    # Display the WordCloud
    st.subheader("🌟 Resume WordCloud (Most Frequent Words)")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Display extracted keywords
    st.subheader("🔑 Extracted Keywords from Resume")
    st.write(", ".join(keywords[:10]))  # Display top 10 keywords

    # Optionally, show more details based on analysis (like skills, experience, etc.)
    # This section can be further expanded with detailed NLP analysis.
