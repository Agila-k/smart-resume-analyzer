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
# ... [rest of the code above remains unchanged]

# Custom CSS (add slightly darker background so it doesn't look plain)
st.markdown("""
    <style>
        .upload-section {
            margin-top: 20px;
            padding: 20px;
            background-color: #f3f3f3;
            border-radius: 8px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.06);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">📄 Smart Resume Analyzer using NLP</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-name">Developed by <b>Agila Karunanithi</b></div>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Show Upload section only if checkbox is ticked
if st.checkbox("Start Resume Analysis"):
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
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
