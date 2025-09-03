# ml_model.py
import numpy as np

def predict_signal(df):
    prices = df['price'].values
    slope = (prices[-1] - prices[-10]) / (np.mean(prices[-10:]) + 1e-6) if len(prices) > 10 else 0
    rsi = df['rsi14'].iloc[-1] if 'rsi14' in df.columns else 50
    conf = 0.5
    pred = 'HOLD'
    if slope > 0.005 and rsi < 70:
        pred = 'BUY'; conf = 0.6 + min(0.4, slope*20)
    elif slope < -0.005 and rsi > 30:
        pred = 'SELL'; conf = 0.6 + min(0.4, -slope*20)
    return pred, conf
