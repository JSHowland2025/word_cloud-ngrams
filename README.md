# N-Gram Word Cloud — Full Feature Version

This Streamlit application generates word clouds from n-grams extracted from text stored in a CSV file. It supports mask-based shapes, sentiment coloring, satisfaction scoring, JEA brand palettes, custom fonts, stopword editing, and multiple export formats (PNG, high-resolution PNG, PDF, and SVG).

## Features

- Upload a CSV file and select the column containing the text
- Choose n-gram size (1–3), minimum frequency, and maximum number of n-grams
- Optional lemmatization of tokens
- Stopword customization:
  - Built-in NLTK English stopwords
  - Additional stopwords read from `stopwords.txt` if present
  - UI field to add custom stopwords during a session
  - Button to reset stopwords to defaults
- Mask support for shaped layouts:
  - PNG/JPG mask upload
  - Optional mask inversion
  - Optional soft edge via Gaussian blur
  - Separate outside/inside fill colors
- Colorization choices:
  - Default Matplotlib colormaps
  - JEA brand palette selection
  - VADER sentiment-based coloring
  - Satisfaction-based numeric sentiment coloring
- Font selection via bundled `.ttf` files
- Download outputs:
  - PNG
  - High-resolution PNG (600 DPI)
  - PDF
  - SVG
- Optional frequency table with highlighted contextual examples and CSV export

## Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

