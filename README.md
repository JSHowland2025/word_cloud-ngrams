# 📊 N-Gram Word Cloud Generator

This is a Python web app built with [Streamlit](https://streamlit.io) that allows you to:
- Upload a CSV file
- Extract and visualize n-grams as a word cloud
- View and download n-gram frequency tables

---

## 🚀 Features

- Supports unigrams, bigrams, and trigrams (n-grams)
- Optional lemmatization of text
- Custom word cloud title
- Adjustable word cloud size and frequency filtering
- Downloadable word cloud image and CSV frequency table

---

## 📁 Files Included

- `ngram_wordcloud_app.py` — The main Streamlit app
- `stopwords.txt` — Custom stopword list (one word per line)
- `requirements.txt` — List of Python packages to install
- `speech_bubble_icon.png` *(optional)* — App icon for the browser tab

---

## 🧰 Requirements

- Python 3.7 or newer
- pip (Python package manager)
- Internet access (for downloading NLTK resources)

---

## ⚙️ Installation

1. Clone or download the project folder.

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download necessary NLTK data (first time only):

   ```bash
   python
   >>> import nltk
   >>> nltk.download('punkt')
   >>> nltk.download('wordnet')
   >>> exit()
   ```

---

## ▶️ Running the App

Run the app from your terminal or command prompt:

```bash
streamlit run ngram_wordcloud_app.py
```

Your browser will open to: `http://localhost:8501`

---

## 📤 Sharing Options

### Option 1: Share Static Outputs
You can export:
- Word cloud PNG files
- N-gram frequency CSV files  
And upload them to SharePoint or email them to colleagues.

### Option 2: Deploy Online (optional)
Deploy to [Streamlit Cloud](https://streamlit.io/cloud) or similar, then:
- Share the public URL via SharePoint or email

---

## 💡 Notes

- The CSV you upload must contain at least one column with natural language text.
- The stopword list can be customized by editing `stopwords.txt`.

---

## 📬 Contact

For questions, bug reports, or contributions, feel free to open an issue or get in touch.
