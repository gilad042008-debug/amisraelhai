# utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io


def extract_chart_area(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0,0,img.shape[1],img.shape[0]
    areas = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours]
    areas.sort(key=lambda x: x[1], reverse=True)
    x,y,w,h = areas[0][0]
    pad = 6
    return max(0,x-pad), max(0,y-pad), min(img.shape[1]-x, w+pad), min(img.shape[0]-y, h+pad)


def extract_series_from_chart(img, rect):
    x,y,w,h = rect
    chart = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    if np.mean(blurred) < 127:
        blurred = cv2.bitwise_not(blurred)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    cols = []
    for col in range(norm.shape[1]):
        colvals = norm[:, col]
        idx = np.where(colvals < 150)[0]
        cols.append(np.median(idx) if idx.size else np.nan)
    arr = np.array(cols)
    nans = np.isnan(arr)
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
    series_norm = 1 - (arr / norm.shape[0])
    return series_norm, chart


def map_pixels_to_prices(series_norm, ocr_nums):
    if len(ocr_nums) >= 2:
        top_price, bottom_price = max(ocr_nums), min(ocr_nums)
        prices = bottom_price + series_norm * (top_price - bottom_price)
        confidence = 0.9
    else:
        prices = 100 * (series_norm - series_norm.min()) / (series_norm.max() - series_norm.min()) + 10
        confidence = 0.4
    return prices, confidence


def annotate_chart(chart_crop, df, entry, stop, targets, decision):
    pil = Image.fromarray(cv2.cvtColor(chart_crop, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    w,h = pil.size
    pmin, pmax = df['price'].min(), df['price'].max()
    def price_to_y(p):
        return int((1 - (p - pmin) / (pmax - pmin)) * h)
    # entry
    draw.line([(0, price_to_y(entry)), (w, price_to_y(entry))], fill='orange', width=3)
    draw.text((10, price_to_y(entry)-15), f'Entry {entry}', fill='orange')
    # stop
    draw.line([(0, price_to_y(stop)), (w, price_to_y(stop))], fill='red', width=3)
    draw.text((10, price_to_y(stop)-15), f'Stop {stop}', fill='red')
    # targets
    for i,t in enumerate(targets):
        draw.line([(0, price_to_y(t)), (w, price_to_y(t))], fill='green', width=2)
        draw.text((10, price_to_y(t)-15), f'TP{i+1} {t}', fill='green')
    draw.text((10,10), f'Decision: {decision}', fill='white')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return buf.getvalue()
