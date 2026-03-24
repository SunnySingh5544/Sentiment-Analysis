# ViewPoint — YouTube Sentiment Analyser
# Your Complete Project

## Folder Structure
```
final-app/
├── backend/
│   ├── app.py                  ← Flask server
│   ├── requirements.txt        ← Python packages
│   ├── sentiment_model.pkl     ← YOUR trained model (copy here)
│   └── tfidf_vectorizer.pkl    ← YOUR vectorizer (copy here)
└── frontend/
    └── index.html              ← Open this in browser
```

## Setup in 4 Steps

### Step 1 — Copy your trained model files
Copy these 2 files from your Downloads into the backend/ folder:
- sentiment_model.pkl
- tfidf_vectorizer.pkl

### Step 2 — Add your YouTube API key
Open backend/app.py
Find line: API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"
Replace with your actual key

### Step 3 — Install and run backend
Open terminal in the backend/ folder:
pip install -r requirements.txt
python app.py

You should see: "Model loaded successfully!" and "Running on http://127.0.0.1:5000"

### Step 4 — Open frontend
Open frontend/index.html in Chrome
Paste any YouTube video URL
Click Analyse
See your model's predictions!