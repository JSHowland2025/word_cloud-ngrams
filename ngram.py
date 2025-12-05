# =============================================================================
# N-Gram Word Cloud — Full Feature Version with PII Scrubber
# Mask-aware background, Soft/No edge, unified color options for inside/outside,
# sentiment coloring (VADER or satisfaction), token-aware highlighting, downloads
# svg output
# =============================================================================
# JS Howland 12/05/2025

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
from wordcloud import WordCloud
import nltk
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords as nltk_stopwords
import matplotlib.pyplot as plt
from matplotlib import colors
import xml.etree.ElementTree as ET
import os


# ---------------------- One-time NLTK setup ----------------------
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

# ---------------------- App config ----------------------
st.set_page_config(page_title="N-Gram Word Cloud (Full)", layout="centered")

# ---------------------- JEA Fonts (Google equivalents) ----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500&family=Libre+Franklin:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Libre Franklin', sans-serif;
}
h1, h2, h3 {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Color definitions ----------------------
DEFAULT_COLORMAPS = ["viridis", "plasma", "cividis", "Pastel1", "Set2", "cubehelix"]
BG_COLOR_OPTIONS = [
    "white", "ghostwhite", "mintcream", "ivory",
    "lavenderblush", "honeydew", "aliceblue", "beige",
    "floralwhite", "seashell", "linen", "azure"
]
INSIDE_COLOR_OPTIONS = BG_COLOR_OPTIONS

SENTIMENT_PALETTES = [
    "Red - Grey - Green",
    "Orange - White - Blue",
    "Purple - Grey - Teal",
]

# ---------------------- JEA Brand Color Palette ----------------------
#JEA_COLORS = {
#    "JEA Brand Blue": "#002D72",
#    "JEA Gold": "#FFC72C",
#    "JEA Orange": "#E35205",
#    "JEA Sky Blue": "#0072CE",
#    "JEA Teal": "#007681",
#    "Bright Red": "#E03C31",
#    "Violet": "#4B306A", 
#    "Gray": "#A7A8AA",
#    "Aqua": "#5BC2E7",
#    "Lime": "#B7BF10",
#   "Sea Blue": "#005F83",
#    "Purple": "#5E2751"
#}

JEA_COLORS = {
    "JEA Brand Blue": "#012169",
    "JEA_blue_1": "#1D4F91",
    "JEA_blue_2": "#3ba5d4",
    "JEA_blue_3": "#4EC3E0"    
}


# ---------------------- Setup ----------------------
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

base_stop_words = set(nltk_stopwords.words("english"))
try:
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        base_stop_words |= {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    pass

if 'stop_words' not in st.session_state:
    st.session_state.stop_words = base_stop_words.copy()

# ---------------------- Helpers ----------------------
def clean_tokens(text, lemmatize):
    text = text.replace("’", "'").lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in st.session_state.stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def extract_ngram_list(tokens, n):
    return [' '.join(g) for g in ngrams(tokens, n)]

def color_from_sentiment(score, palette):
    norm = (score + 1) / 2
    if palette == "Red - Grey - Green":
        neg, neu, pos = '#FF0000', '#808080', '#00FF00'
    elif palette == "Orange - White - Blue":
        neg, neu, pos = '#FFA500', '#FFFFFF', '#1E90FF'
    else:
        neg, neu, pos = '#800080', '#A9A9A9', '#008080'
    neg, neu, pos = map(np.array, map(colors.to_rgb, [neg, neu, pos]))
    if norm < 0.5:
        mix = norm * 2
        rgb = neg * (1 - mix) + neu * mix
    else:
        mix = (norm - 0.5) * 2
        rgb = neu * (1 - mix) + pos * mix
    return colors.to_hex(rgb)

def aggregate_scores(scores, method):
    if method == "Median":
        return float(np.median(scores))
    if method == "Mode":
        s = pd.Series(scores)
        return float(s.mode().iloc[0]) if not s.mode().empty else float(np.mean(scores))
    return float(np.mean(scores))

def normalize_satisfaction(series, fill_method):
    if not pd.api.types.is_numeric_dtype(series):
        return None, "Satisfaction score column must be numeric."
    if series.dropna().min() < 0:
        return None, "Satisfaction scores must be >= 0."
    min_val, max_val = series.min(), series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return None, "Satisfaction score has no usable range."
    nonnull = series.dropna()
    if nonnull.empty:
        return None, "Satisfaction score column is entirely missing."
    if fill_method == "Median":
        fill_value = nonnull.median()
    elif fill_method == "Mode":
        m = nonnull.mode()
        fill_value = m.iloc[0] if not m.empty else nonnull.mean()
    else:
        fill_value = nonnull.mean()
    filled = series.fillna(fill_value)
    norm = 2 * ((filled - min_val) / (max_val - min_val)) - 1
    return norm, None

# ---------------------- UI: Uploads ----------------------
st.title("N-Gram Word Cloud — Full App")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
mask_file = st.file_uploader("Optional: Upload a mask image (PNG/JPG)", type=["png", "jpg"])

if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

invert_mask = False
if mask_file:
    invert_mask = st.checkbox("Invert mask (draw inside dark areas)", value=True)

# ---------------------- Auto-detect text columns ----------------------
text_like_cols = [c for c in df.columns if (df[c].dtype == object) or (df[c].astype(str).str.len().mean() > 10)]
if not text_like_cols:
    text_like_cols = list(df.columns)
text_col = st.selectbox("Select the text column:", text_like_cols)
if not pd.api.types.is_string_dtype(df[text_col]):
    df[text_col] = df[text_col].astype(str)

# ---------------------- N-gram controls ----------------------
cols_top = st.columns(3)
with cols_top[0]:
    n = st.slider("N-gram size", 1, 3, 2)
with cols_top[1]:
    max_words = st.slider("Max n-grams", 10, 300, 100, step=10)
with cols_top[2]:
    min_freq = st.slider("Min frequency", 1, 10, 1)
lemmatize = st.checkbox("Apply lemmatization", value=False)

# ---------------------- Stopword customization ----------------------
st.markdown("---")
st.subheader("Stopword customization")
st.caption(f"Base stopwords size: {len(base_stop_words)}")

custom_sw = st.text_area("Add custom stopwords (comma or space separated)")
if custom_sw:
    extra = {w.strip().lower() for w in re.split(r"[\s,]+", custom_sw) if w.strip()}
    st.session_state.stop_words.update(extra)
    st.caption(f"Custom stopwords added: {len(extra)} (Total: {len(st.session_state.stop_words)})")
else:
    st.caption(f"Total stopwords in use: {len(st.session_state.stop_words)}")

if st.button("Reset stopwords to default"):
    st.session_state.stop_words = base_stop_words.copy()
    st.success("Stopwords reset to default.")

# ---------------------- Visualization controls ----------------------
st.markdown("---")
st.subheader("Colors for masked layout")

combined_colors = {**JEA_COLORS, **{c.capitalize(): c for c in BG_COLOR_OPTIONS}}

col_v1, col_v2 = st.columns(2)
with col_v1:
    color_mode = st.radio(
        "Colorization method",
        ["Default colormap", "App-generated sentiment (VADER)", "Satisfaction-based sentiment"],
        index=0,
    )
with col_v2:
    page_bg_name = st.selectbox("Background color (outside mask)", list(combined_colors.keys()), index=0)
    page_bg_color = combined_colors[page_bg_name]

mask_edge = None
blur_radius = 0
use_fill_color = False
fill_color = None
if mask_file:
    mask_edge = st.radio("Mask edge style", ["No edge", "Soft edge"], index=0)
    blur_radius = st.slider("Soft edge radius (px)", 0, 6, 2) if mask_edge == "Soft edge" else 0
    use_fill_color = st.checkbox("Use a separate color inside the shape", value=True)
    if use_fill_color:
        fill_name = st.selectbox("Fill color (inside mask)", list(combined_colors.keys()), index=1)
        fill_color = combined_colors[fill_name]

colormap = None
sent_palette = None
agg_method = "Mean"
score_col = None

# Optional JEA word palette
if color_mode == "Default colormap":
    use_jea_palette = st.checkbox("Use JEA brand palette for word colors", value=False)
    if use_jea_palette:
        word_palette = st.multiselect("Select JEA colors for words", list(JEA_COLORS.keys()),
                                      default=["JEA Brand Blue","JEA_blue_1","JEA_blue_2","JEA_blue_3"])
        selected_hex = [JEA_COLORS[name] for name in word_palette]
        def jea_color_func(word, **kwargs):
            return np.random.choice(selected_hex)
        colormap = None
    else:
        colormap = st.selectbox("Colormap", DEFAULT_COLORMAPS, index=0)
else:
    sent_palette = st.selectbox("Sentiment palette", SENTIMENT_PALETTES, index=0)
    if color_mode == "Satisfaction-based sentiment":
        score_col = st.selectbox("Select satisfaction score column", df.columns)
        agg_method = st.selectbox("Aggregation method for n-gram sentiment", ["Mean", "Median", "Mode"], index=0)

# --------------------Font selections --------------------
FONT_OPTIONS = {
    "Franklin Gothic": "fonts/FRABK.ttf",
    "Libre Franklin": "fonts/LibreFranklin-Regular.ttf",
    "Oswald": "fonts/Oswald-Regular.ttf",
    "Lato (Polish)": "fonts/Lato-Regular.ttf",
}

# Optional: a tiny helper to namespace keys in this section
def viz_key(name: str) -> str:
    return f"viz__{name}"

# Warn if missing files (for debugging, shows absolute paths)
missing = []
for name, rel in FONT_OPTIONS.items():
    if not os.path.exists(rel):
        missing.append(f"{name}: {os.path.abspath(rel)}")
if missing:
    st.warning("Missing font files:\n" + "\n".join(f"- {m}" for m in missing))

font_choice = st.selectbox(
    "Font for word cloud text",
    list(FONT_OPTIONS.keys()),
    index=1,
    key=viz_key("font_select"),   # <-- unique key
)
font_path = FONT_OPTIONS[font_choice]
if not os.path.exists(font_path):
    font_path = None  # fallback to default


# ---------------------- Processing ----------------------
st.markdown("---")
with st.spinner("Processing text..."):
    mask = None
    mask_img = None
    if mask_file:
        mask_img = Image.open(mask_file).convert("L")
        if invert_mask:
            mask_img = ImageOps.invert(mask_img)
        mask = np.array(mask_img)

    sia = SentimentIntensityAnalyzer() if color_mode == "App-generated sentiment (VADER)" else None
    row_scores = None
    if color_mode == "Satisfaction-based sentiment":
        norm, err = normalize_satisfaction(df[score_col], fill_method=agg_method)
        if err:
            st.error(err)
            st.stop()
        row_scores = norm

    ngram_freqs = Counter()
    ngram_scores = {}
    texts = df[text_col].astype(str).tolist()
    for i, comment in enumerate(texts):
        tokens = clean_tokens(comment, lemmatize)
        ng_list = extract_ngram_list(tokens, n)
        if not ng_list:
            continue
        if color_mode == "App-generated sentiment (VADER)":
            score = float(sia.polarity_scores(comment)['compound'])
        elif color_mode == "Satisfaction-based sentiment":
            score = float(row_scores.iloc[i])
        else:
            score = None
        for ng in ng_list:
            ngram_freqs[ng] += 1
            if score is not None:
                ngram_scores.setdefault(ng, []).append(score)

    filtered_freqs = {k: v for k, v in ngram_freqs.items() if v >= min_freq}
    filtered_freqs = dict(sorted(filtered_freqs.items(), key=lambda x: x[1], reverse=True)[:max_words])

    ngram_sentiment_value = None
    if color_mode == "App-generated sentiment (VADER)":
        ngram_sentiment_value = {k: float(np.mean(ngram_scores.get(k, [0.0]))) for k in filtered_freqs}
    elif color_mode == "Satisfaction-based sentiment":
        ngram_sentiment_value = {k: aggregate_scores(ngram_scores.get(k, [0.0]), agg_method) for k in filtered_freqs}

    wc = WordCloud(
        width=900, height=650,
        background_color=None if mask_file else page_bg_color,
        mode='RGBA' if mask_file else 'RGB',
        collocations=False,
        mask=mask,
        font_path = font_path

    ).generate_from_frequencies(filtered_freqs)

    if color_mode == "Default colormap":
        if 'use_jea_palette' in locals() and use_jea_palette:
            wc_image = wc.recolor(color_func=jea_color_func).to_image()
        else:
            wc_image = wc.recolor(colormap=colormap).to_image()
    else:
        def color_func(word, **kwargs):
            score = ngram_sentiment_value.get(word, 0.0) if ngram_sentiment_value else 0.0
            return color_from_sentiment(score, sent_palette)
        wc_image = wc.recolor(color_func=color_func).to_image()

    if mask_file:
        base = Image.new("RGBA", wc_image.size, page_bg_color)
        base.putalpha(255)
        alpha_img = mask_img
        if mask_edge == "Soft edge" and blur_radius > 0:
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        alpha = np.array(alpha_img, dtype=np.uint8)
        if use_fill_color and fill_color:
            fill_layer = Image.new("RGBA", wc_image.size, fill_color)
            fill_layer.putalpha(Image.fromarray(alpha, mode="L"))
            base = Image.alpha_composite(base, fill_layer)
        wc_rgba = wc_image.convert("RGBA")
        wc_image = Image.alpha_composite(base, wc_rgba)

# ---------------------- SVG Export ----------------------
svg_io = BytesIO()
svg_root = ET.Element('svg', xmlns="http://www.w3.org/2000/svg",
                      width=str(wc.width), height=str(wc.height))

for (word, freq), font_size, position, orientation, color in wc.layout_:
    if word is None:
        continue
    x, y = position
    text_el = ET.SubElement(
        svg_root, 'text',
        x=str(x),
        y=str(y),
        fill=color or "#000000",
        style=(
            f"font-size:{font_size}px;"
            "font-family:'Libre Franklin','Oswald',Arial,sans-serif;"
        )
    )
    # Rotation if orientation is vertical
    if orientation != None and orientation != 0:
        text_el.set("transform", f"rotate({orientation},{x},{y})")
    text_el.text = word

svg_data = ET.tostring(svg_root, encoding='utf-8', method='xml')
svg_io.write(svg_data)
svg_bytes = svg_io.getvalue()



# ---------------------- Display + download ----------------------
st.markdown("### Word Cloud")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(wc_image, caption=f"{n}-gram Word Cloud", width=700)

img_io = BytesIO()
wc_image.save(img_io, format='PNG')
st.download_button("Download Word Cloud (PNG)", data=img_io.getvalue(), file_name="wordcloud.png", mime="image/png")

# ---------------------- High-Resolution PNG Export (preserves mask/background) ----------------------
highres_io = BytesIO()
scale_factor = 4  # 4x resolution, adjust as needed

# Use the fully composited RGBA image (wc_image), not wc.to_image()
wc_highres = wc_image.resize(
    (wc_image.width * scale_factor, wc_image.height * scale_factor),
    Image.LANCZOS
)
wc_highres.save(highres_io, format="PNG", dpi=(600, 600))

st.download_button(
    "Download Word Cloud (High-Res PNG, 600 DPI)",
    data=highres_io.getvalue(),
    file_name="wordcloud_highres.png",
    mime="image/png"
)

# ---------------------- PDF Export (Matplotlib vector surface) ----------------------
pdf_io = BytesIO()

import matplotlib.pyplot as plt

# Use Matplotlib to render the final image to a PDF surface.
# This keeps internal vector text handling and high-quality scaling.
fig, ax = plt.subplots(
    figsize=(wc_image.width / 100, wc_image.height / 100),
    dpi=100
)
ax.imshow(wc_image)
ax.axis("off")
plt.tight_layout(pad=0)

plt.savefig(pdf_io, format="pdf", bbox_inches="tight", pad_inches=0)
plt.close(fig)

st.download_button(
    "Download Word Cloud (PDF)",
    data=pdf_io.getvalue(),
    file_name="wordcloud.pdf",
    mime="application/pdf",
)

# --------SVG download ---------------------------------------
st.download_button("Download Word Cloud (SVG)", data=svg_bytes,
                   file_name="wordcloud.svg", mime="image/svg+xml")


# ---------------------- Frequency table ----------------------
show_table = st.checkbox("Show frequency table with highlighted examples")
if show_table:
    df_freq = pd.DataFrame(list(filtered_freqs.items()), columns=["N-gram", "Frequency"])
    original_texts = df[text_col].astype(str).tolist()
    raw_token_rows, cleaned_token_rows, clean_to_raw_maps = [], [], []
    for text in original_texts:
        raw_tokens = tokenizer.tokenize(text)
        raw_token_rows.append(raw_tokens)
        cleaned_tokens, mapping = [], []
        for idx, tok in enumerate(raw_tokens):
            t = tok.lower()
            if t in st.session_state.stop_words:
                continue
            if lemmatize:
                t = lemmatizer.lemmatize(t)
            cleaned_tokens.append(t)
            mapping.append(idx)
        cleaned_token_rows.append(cleaned_tokens)
        clean_to_raw_maps.append(mapping)
    example_map = {}
    for ngram_text in df_freq["N-gram"]:
        parts = ngram_text.split()
        found = False
        for row_i, c_tokens in enumerate(cleaned_token_rows):
            for j in range(len(c_tokens) - len(parts) + 1):
                if c_tokens[j:j+len(parts)] == parts:
                    raw_tokens = raw_token_rows[row_i]
                    raw_idx = clean_to_raw_maps[row_i][j:j+len(parts)]
                    highlighted = " ".join(
                        f"<span style='background-color: rgba(255,255,0,0.4);'>{tok}</span>" if k in raw_idx else tok
                        for k, tok in enumerate(raw_tokens)
                    )
                    example_map[ngram_text] = highlighted
                    found = True
                    break
            if found:
                break
        if not found:
            example_map[ngram_text] = "No example found"
    df_freq["Example"] = df_freq["N-gram"].map(example_map)
    if color_mode != "Default colormap" and ngram_sentiment_value is not None:
        df_freq["SentimentScore"] = df_freq["N-gram"].map(ngram_sentiment_value)
    df_freq = df_freq.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    st.write(df_freq.to_html(escape=False), unsafe_allow_html=True)
    csv_df = df_freq.copy()
    csv_df["Example"] = csv_df["Example"].str.replace(r"<.*?>", "", regex=True)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Frequency Table (CSV)", data=csv_bytes, file_name="ngram_frequencies.csv", mime="text/csv")
