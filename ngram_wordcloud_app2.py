# J Howland 2025
# Create a wordcloud with n-grams
# Updated 072925

import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from io import BytesIO
from PIL import Image
import nltk
import random

# Ensure necessary NLTK data is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.data.path.append("nltk_data")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="N-Gram Word Cloud",
    page_icon="speech_bubble_icon.png",
    layout="centered"
)

# --- Styles ---
st.markdown("""
    <style>
    html, body, .css-10trblm, .stTextInput, .stSelectbox, .stSlider, .css-1v0mbdj, .css-1d391kg {
        font-size: 20px !important;
    }
    .css-1aumxhk {
        font-size: 30px !important;
    }
    .stSlider > div > div {
        height: 35px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Stopwords from Local File ---
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stop_words = set(line.strip() for line in f if line.strip())

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# --- Available Color Schemes & Default Background ---
COLOR_SCHEMES = [
    "viridis",   # default WordCloud colormap
    "plasma",
    "cividis",
    "Pastel1",
    "Set2",
    "cubehelix"
]
DEFAULT_BG = "#FFFFFF"  # white

# --- Text Cleaner ---
def clean_text(text, apply_lemmatization=False):
    text = text.replace("’", "'").lower()
    tokens = tokenizer.tokenize(text)
    cleaned = [t for t in tokens if t not in stop_words]
    if apply_lemmatization:
        cleaned = [lemmatizer.lemmatize(t) for t in cleaned]
    return cleaned

# --- N-Gram Extractor ---
def extract_ngrams(tokens, n):
    return Counter([' '.join(gram) for gram in ngrams(tokens, n)])

# --- Word Cloud Generator ---
def generate_wordcloud(freqs, colormap, bg_color):
    wc = WordCloud(
        width=800,
        height=600,
        background_color=bg_color,
        collocations=False,
        colormap=colormap
    )
    return wc.generate_from_frequencies(freqs)

# --- App UI ---
st.title("N-Gram Word Cloud Generator")
uploaded_file = st.file_uploader("\U0001F4C2 Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("File uploaded is empty.")
        st.stop()

    st.success("File uploaded successfully.")

    col = st.selectbox("Select the text column to analyze:", df.columns)

#    if not pd.api.types.is_string_dtype(df[col]):
#        st.error("Selected column must contain text.")
#        st.stop()

    if not pd.api.types.is_string_dtype(df[col]):
        try:
            df[col] = df[col].astype(str)
            st.success("✅ Column was not originally text but has been successfully converted to string.")
        except Exception as e:
            st.error(f"❌ Could not convert column to string: {e}")
            st.stop()

    n = st.slider("Choose n-gram size", min_value=1, max_value=3, value=2)
    max_words = st.slider("Max number of n-grams to display", min_value=10, max_value=300, value=100, step=10)
    min_freq = st.slider("Minimum frequency threshold", min_value=1, max_value=10, value=1)
    
    lemmatize = st.checkbox(
    "Apply lemmatization",
    value=False,
    help="Lemmatization reduces words to their dictionary form (e.g., 'running' → 'run')."
    )

    #lemmatize = st.checkbox("Apply lemmatization", value=False)

    # --- Visualization Options ---
    color_scheme = st.selectbox("Color scheme", COLOR_SCHEMES, index=0)
    bg_color = st.color_picker("Background color", DEFAULT_BG)

    # --- Additional Stop Words ---
    extra_stopwords_text = st.text_area("Additional stop words (comma or space separated)")
    extra_stopwords = {w.strip().lower() for w in re.split(r'[,\s]+', extra_stopwords_text) if w.strip()}
    stop_words.update(extra_stopwords)

    title = st.text_input("Word Cloud Title (optional)")

    st.caption(f"Stopwords loaded: {len(stop_words)} (including user additions)")

    with st.spinner("Processing text..."):
        text_blob = " ".join(df[col].astype(str).tolist())
        tokens = clean_text(text_blob, apply_lemmatization=lemmatize)
        ngram_freqs = extract_ngrams(tokens, n)
        filtered_freqs = {k: v for k, v in ngram_freqs.items() if v >= min_freq}
        filtered_freqs = dict(sorted(filtered_freqs.items(), key=lambda x: x[1], reverse=True)[:max_words])
        wc = generate_wordcloud(filtered_freqs, color_scheme, bg_color)

    # Display Word Cloud
    st.markdown("### Word Cloud:")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(wc.to_image(), caption=f"{n}-gram Word Cloud" + (f" - {title}" if title else ""), width=600)

    # PNG Download
    img_io = BytesIO()
    wc.to_image().save(img_io, format='PNG')
    st.download_button(
        label="Download Word Cloud as PNG",
        data=img_io.getvalue(),
        file_name="wordcloud.png",
        mime="image/png"
    )

    # Frequency Table and CSV Download
    if st.checkbox("Show raw frequency table"):
        df_freq = pd.DataFrame(filtered_freqs.items(), columns=["N-gram", "Frequency"])


        # Build a mapping from ngram to one example sentence
        example_map = {}
        all_sentences = df[col].astype(str).tolist()

# -----------------NEW---07292025------------------

        # Tokenize and store cleaned tokens for each sentence
        tokenized_rows = [clean_text(text, apply_lemmatization=lemmatize) for text in df[col].astype(str).tolist()]
        original_texts = df[col].astype(str).tolist()

        example_map = {}

        for ngram in filtered_freqs.keys():
            ngram_tokens = ngram.split()
            match_idxs = [
                idx for idx, tokens in enumerate(tokenized_rows)
                if any(' '.join(tokens[i:i+len(ngram_tokens)]) == ngram for i in range(len(tokens)-len(ngram_tokens)+1))
            ]
            if match_idxs:
                chosen_idx = random.choice(match_idxs)
                example_map[ngram] = original_texts[chosen_idx]
            else:
                example_map[ngram] = "No example found"


# -----------------------------------------------

        df_freq = df_freq.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
        st.dataframe(df_freq)

        csv_data = df_freq.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Frequency Table as CSV",
            data=csv_data,
            file_name="ngram_frequencies.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file to get started.")
