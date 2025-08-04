# =============================================================================
# J Howland 2025 - Sentiment-Colorized N-Gram Word Cloud
# Updated 07292025
# =============================================================================

# =============================================================================
# 1. Imports and NLTK Setup
# =============================================================================

import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from io import BytesIO
from PIL import Image
import numpy as np
from matplotlib import colors
import nltk

# --- Ensure NLTK resources only once per session ---
@st.cache_resource
def ensure_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'vader_lexicon': 'sentiment/vader_lexicon',
        'stopwords': 'corpora/stopwords'
    }
    for key, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(key)

ensure_nltk_resources()

# =============================================================================
# 2. Page Configuration and Styling
# =============================================================================

st.set_page_config(
    page_title="N-Gram Word Cloud",
    page_icon="speech_bubble_icon.png",
    layout="centered"
)

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

# =============================================================================
# 3. Stopwords and Tools Initialization
# =============================================================================

with open("stopwords.txt", "r", encoding="utf-8") as f:
    base_stop_words = set(line.strip().lower() for line in f if line.strip())

if 'stop_words' not in st.session_state:
    st.session_state.stop_words = base_stop_words.copy()

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

# =============================================================================
# 4. Helper Functions
# =============================================================================

def clean_text(text, apply_lemmatization=False):
    text = text.replace("’", "'").lower()
    tokens = tokenizer.tokenize(text)
    cleaned = [t for t in tokens if t not in st.session_state.stop_words]
    if apply_lemmatization:
        cleaned = [lemmatizer.lemmatize(t) for t in cleaned]
    return cleaned

def sentiment_to_color(score, palette):
    norm_score = (score + 1) / 2
    if palette == "Red - Grey - Green":
        neg, neu, pos = '#FF0000', '#808080', '#00FF00'
    elif palette == "Orange - White - Blue":
        neg, neu, pos = '#FFA500', '#FFFFFF', '#1E90FF'
    elif palette == "Purple - Grey - Teal":
        neg, neu, pos = '#800080', '#A9A9A9', '#008080'

    neg = np.array(colors.to_rgb(neg))
    neu = np.array(colors.to_rgb(neu))
    pos = np.array(colors.to_rgb(pos))

    if norm_score < 0.5:
        mix = norm_score * 2
        color_rgb = neg * (1 - mix) + neu * mix
    else:
        mix = (norm_score - 0.5) * 2
        color_rgb = neu * (1 - mix) + pos * mix

    return colors.to_hex(color_rgb)

# =============================================================================
# 5. App UI and Main Logic
# =============================================================================

st.title("Sentiment-Colorized N-Gram Word Cloud")

uploaded_file = st.file_uploader("\U0001F4C2 Upload a CSV file", type="csv")
mask_file = st.file_uploader("Optional: Upload a PNG/JPG mask for shaping the word cloud", type=["png", "jpg"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.empty:
        st.error("❌ Uploaded file is empty.")
        st.stop()

    st.success("✅ File uploaded successfully.")
    col = st.selectbox("Select the text column:", df.columns)

    message_slot = st.empty()
    if not pd.api.types.is_string_dtype(df[col]):
        message_slot.warning("⚠️ Selected column is not recognized as text. Attempting to convert to string.")
        try:
            df[col] = df[col].astype(str)
            message_slot.success("✅ Column converted to string.")
        except Exception as e:
            message_slot.error(f"❌ Could not convert column: {e}")
            st.stop()

    # --- Main Controls ---
    n = st.slider("🔢 Choose n-gram size", 1, 3, 2)
    max_words = st.slider("🔢 Max number of n-grams", 10, 300, 100, step=10)
    min_freq = st.slider("🔽 Minimum frequency", 1, 10, 1)
    lemmatize = st.checkbox("Apply lemmatization", value=False,
        help="Lemmatization reduces words to their dictionary form (e.g., 'running' → 'run').")
    palette = st.selectbox("Choose color palette:", ["Red - Grey - Green", "Orange - White - Blue", "Purple - Grey - Teal"])
    bg_color = st.selectbox("Background color:", ["white", "lightgrey", "beige", "lavender", "mintcream"])
    sentiment_filter = st.selectbox("🔍 Filter by sentiment:", ["All", "Positive", "Neutral", "Negative"])
    title = st.text_input("Title (optional)")

    # --- Stopword Controls ---
    st.markdown("---")
    st.subheader("Stopword Customization")
    st.caption(f"Base stopwords loaded: {len(base_stop_words)}")

    custom_stopwords_input = st.text_area("➕ Add custom stopwords (comma-separated):")
    if custom_stopwords_input:
        custom_stopwords = {word.strip().lower() for word in custom_stopwords_input.split(",") if word.strip()}
        st.session_state.stop_words.update(custom_stopwords)
        st.caption(f"Custom stopwords added: {len(custom_stopwords)} (Total now: {len(st.session_state.stop_words)})")
    else:
        st.caption(f"Total stopwords in use: {len(st.session_state.stop_words)}")

    if st.button("🔁 Reset custom stopwords to default"):
        st.session_state.stop_words = base_stop_words.copy()
        st.success("Stopwords reset to base list.")

    # =============================================================================
    # 6. Sentiment Analysis and Word Cloud Generation
    # =============================================================================

    with st.spinner("🔄 Processing text and sentiment..."):
        sia = SentimentIntensityAnalyzer()
        df['SentimentScore'] = df[col].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])

        ngram_sentiments = {}
        ngram_counts = Counter()

        for comment, score in zip(df[col].astype(str), df['SentimentScore']):
            tokens = clean_text(comment, apply_lemmatization=lemmatize)
            ngrams_list = [' '.join(g) for g in ngrams(tokens, n)]
            for ng in ngrams_list:
                ngram_counts[ng] += 1
                ngram_sentiments.setdefault(ng, []).append(score)

        filtered_freqs = {k: v for k, v in ngram_counts.items() if v >= min_freq}
        filtered_freqs = dict(sorted(filtered_freqs.items(), key=lambda x: x[1], reverse=True)[:max_words])
        ngram_sentiment_avg = {k: sum(ngram_sentiments[k])/len(ngram_sentiments[k]) for k in filtered_freqs}

        if sentiment_filter != "All":
            if sentiment_filter == "Positive":
                filtered_freqs = {k: v for k, v in filtered_freqs.items() if ngram_sentiment_avg.get(k, 0) >= 0.05}
            elif sentiment_filter == "Negative":
                filtered_freqs = {k: v for k, v in filtered_freqs.items() if ngram_sentiment_avg.get(k, 0) <= -0.05}
            elif sentiment_filter == "Neutral":
                filtered_freqs = {k: v for k, v in filtered_freqs.items() if -0.05 < ngram_sentiment_avg.get(k, 0) < 0.05}
            ngram_sentiment_avg = {k: ngram_sentiment_avg[k] for k in filtered_freqs}

        mask = None
        if mask_file:
            try:
                img = Image.open(mask_file).convert("L")
                mask = np.array(img)
            except Exception as e:
                st.error(f"⚠️ Error loading mask: {e}")

        wc = WordCloud(width=800, height=600, background_color=bg_color,
                       collocations=False, mask=mask, contour_width=1, contour_color='black')
        wc.generate_from_frequencies(filtered_freqs)

        def color_func(word, **kwargs):
            score = ngram_sentiment_avg.get(word, 0.0)
            return sentiment_to_color(score, palette)

        colored_img = wc.recolor(color_func=color_func).to_image()

    # =============================================================================
    # 7. Display Results
    # =============================================================================

    st.markdown("### ☁️ Word Cloud Output:")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(colored_img, caption=f"{n}-gram Word Cloud" + (f" - {title}" if title else ""), width=600)

    # --- PNG Download ---
    img_io = BytesIO()
    colored_img.save(img_io, format='PNG')
    st.download_button("⬇️ Download Word Cloud as PNG", img_io.getvalue(), "wordcloud.png", "image/png")

    # --- Frequency Table ---
    if st.checkbox("Show raw frequency table"):
        df_freq = pd.DataFrame(filtered_freqs.items(), columns=["N-gram", "Frequency"])
        df_freq["SentimentScore"] = df_freq["N-gram"].map(ngram_sentiment_avg)
        df_freq = df_freq.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
        st.dataframe(df_freq)

        csv_data = df_freq.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Frequency Table as CSV", csv_data, "ngram_frequencies.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to get started.")
