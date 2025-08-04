# N‑Gram Word Cloud Apps

This repository contains two Streamlit applications for generating n‑gram word clouds from text data in CSV files. One app produces a standard n‑gram word cloud; the other colors each n‑gram by the average sentiment of the comments in which it appears.

## Repository structure

```
.
├── ngram_wordcloud_app2.py
├── sentiment.py
├── stopwords.txt            # user‑maintained list of stopwords (one per line)
├── speech_bubble.ico        # optional app icon (see note below)
├── requirements.txt
└── README.md
```

## Apps at a glance

| App | Purpose | Notable features | Inputs | Outputs |
|---|---|---|---|---|
| `ngram_wordcloud_app2.py` | Generate a standard n‑gram word cloud from a CSV text column | N‑gram size (1–3), min frequency, max words, optional lemmatization, color map selection, background color picker, add extra stopwords at runtime, frequency table + CSV download | CSV with at least one text column | Word cloud PNG download, optional frequency table CSV |
| `sentiment.py` | Generate an n‑gram word cloud with colors determined by average VADER sentiment for each n‑gram | Per‑ngram sentiment coloring (three palettes), optional image mask (PNG/JPG), sentiment filter (All/Positive/Neutral/Negative), background color presets, stopword customization with reset, frequency table includes sentiment | CSV with a text column; optional mask image | Colored word cloud PNG download, frequency table CSV |

### Icon note

Both apps currently set `st.set_page_config(page_icon="speech_bubble_icon.png")`. If you prefer to supply `speech_bubble.ico`, either change the filename in the scripts or add a PNG named `speech_bubble_icon.png` to the repo.

## Installation

1. Ensure Python 3.12 is installed.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### NLTK data

At first run, the apps will download required NLTK resources (e.g., `wordnet`, `punkt`, `vader_lexicon`, `stopwords`) if not already present. No manual setup is required.

## Running the apps

From the repository root:

```bash
# Standard n‑gram word cloud
streamlit run ngram_wordcloud_app2.py

# Sentiment‑colorized n‑gram word cloud
streamlit run sentiment.py
```

Open the local URL printed by Streamlit in your browser.

## Input expectations

- CSV with at least one column containing text.
- If the selected column is not typed as string, both apps attempt to convert it to string.
- `stopwords.txt` is expected to be present in the working directory (one stopword per line, UTF‑8).

## Features

### Common
- N‑gram extraction (1–3).
- Runtime stopword augmentation via text input; counts are based on the combined base list plus user additions.
- Optional lemmatization.
- Exportable outputs: PNG for the word cloud image; CSV for the frequency table.

### `ngram_wordcloud_app2.py`
- Choose color map (e.g., viridis, plasma, cividis, etc.) and any background color via color picker.
- Sliders for minimum frequency threshold and maximum number of n‑grams to render.

### `sentiment.py`
- Average VADER sentiment score per n‑gram used to color the cloud across one of three diverging palettes.
- Optional mask image (PNG/JPG) to shape the cloud.
- Sentiment filter to include only Positive, Neutral, or Negative n‑grams in the visualization.
- Frequency table includes the average sentiment score per n‑gram.

## Notes and tips

- Large CSVs: consider pre‑filtering or sampling to reduce processing time.
- Reproducibility: colors in the standard app come from the selected colormap; the sentiment app deterministically maps sentiment to color on each run.
- Fonts and Unicode: the `wordcloud` library supports basic Unicode; for specialized scripts you may need a custom font (not configured here).

## License

MIT
