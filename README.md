# Streamlit Chart Bot


Ready-to-deploy Streamlit app that analyzes chart screenshots and returns a visual annotated chart plus a short trade recommendation (Buy / Sell / Hold) with entry / stop / target and confidence.


## Features
- Image ingestion (PNG/JPG)
- Pixel->price reconstruction using OCR on axis labels (Tesseract)
- Indicators: SMA20, SMA50, EMA, MACD, RSI
- Rule-based pattern detection (double top/bottom, triangles, h&s skeleton)
- Simple ML classifier (stub) for additional signal confidence
- Annotated chart output (PNG) and downloadable CSV of extracted prices
- Dockerfile + docker-compose for local deployment
- Deployable on Streamlit Cloud & Hugging Face Spaces


## Quick start (local)
1. Install Python 3.11 and Tesseract OCR (system binary).
2. Clone this repo.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`


## Deploy on Streamlit Cloud
1. Push repo to GitHub.
2. Go to https://streamlit.io/cloud and create a new app. Connect your repo and select `app.py`.
3. App auto-deploys.


## Deploy on Hugging Face Spaces
1. Push to GitHub or upload repo to a new Space.
2. Create a new Space and set framework to Streamlit.
3. Upload repo and run.


## Notes
- The ML model is a small placeholder — you can replace `ml_model.py` with a trained model.
- Auto-trading requires you to wire exchange/broker APIs (example skeleton is commented in `utils.py`).
