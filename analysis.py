# analysis.py
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ml_model import predict_signal
from utils import extract_chart_area, extract_series_from_chart, map_pixels_to_prices, annotate_chart
import matplotlib.pyplot as plt


def analyze_image_and_get_plan(img_bytes: bytes):
    # Load image
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1) detect chart area
    rect = extract_chart_area(cv_img)

    # 2) OCR axis labels
    x, y, w, h = rect
    left_strip = cv_img[y:y+h, max(0, x-120):x]
    left_gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY)
    left_gray = cv2.resize(left_gray, None, fx=1.5, fy=1.5)
    _, th = cv2.threshold(left_gray, 200, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(th, config='--psm 6 digits')
    import re
    nums = re.findall(r'[-+]?\d*\.?\d+|\d+', text)
    nums = [float(n) for n in nums] if nums else []

    # 3) extract series from chart
    series_norm, chart_crop = extract_series_from_chart(cv_img, rect)

    # 4) map pixels to prices
    prices, conf_pixels = map_pixels_to_prices(series_norm, nums)

    # build dataframe
    df = pd.DataFrame({'price': prices})
    df['sma20'] = SMAIndicator(df['price'], window=20, fillna=True).sma_indicator()
    df['rsi14'] = RSIIndicator(df['price'], window=14, fillna=True).rsi()

    # 5) basic rule-based signal
    decision = 'HOLD'
    if df['price'].iloc[-1] > df['sma20'].iloc[-1] and df['rsi14'].iloc[-1] < 70:
        decision = 'BUY'
    elif df['price'].iloc[-1] < df['sma20'].iloc[-1] and df['rsi14'].iloc[-1] > 30:
        decision = 'SELL'

    # 6) ML model signal (adds confidence)
    ml_pred, ml_conf = predict_signal(df)
    score = 0.5
    if decision == 'BUY':
        score += 0.2
    elif decision == 'SELL':
        score -= 0.2
    score += (ml_conf - 0.5) * 0.4
    score = max(0, min(1, score))

    # 7) trade plan
    last = df['price'].iloc[-1]
    atr = df['price'].diff().abs().rolling(14).mean().iloc[-1] or (last*0.005)
    entry = round(last, 2)
    stop = round(last - 2*atr, 2)
    targets = [round(last + atr*1.5,2), round(last + atr*3,2)] if decision=='BUY' else [round(last - atr*1.5,2), round(last - atr*3,2)]

    # 8) annotated chart
    annotated_bytes = annotate_chart(chart_crop, df, entry, stop, targets, decision)

    # 9) prepare CSV
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    # 10) indicator figure
    fig = plt.figure(figsize=(8,3))
    plt.plot(df['price'], label='price')
    plt.plot(df['sma20'], label='SMA20')
    plt.legend()

    return {
        'decision': decision,
        'confidence': score,
        'entry': entry,
        'stop': stop,
        'targets': targets,
        'annotated_image_bytes': annotated_bytes,
        'prices_csv': csv_bytes,
        'indicator_fig': fig
    }
