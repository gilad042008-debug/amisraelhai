# Streamlit Chart Bot (app.py)
import streamlit as st
from PIL import Image
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis import analyze_image_and_get_plan

st.set_page_config(page_title='Chart Analyzer Bot', layout='wide')
st.title('📈 Chart Analyzer Bot')

st.markdown('Upload a chart screenshot (line or candlestick). The bot will extract a price series, compute indicators, detect patterns, and give a Buy/Sell/Hold recommendation along with annotated chart and CSV.')

uploaded = st.file_uploader('Upload chart image', type=['png', 'jpg', 'jpeg'])

if uploaded is not None:
    img_bytes = uploaded.read()
    st.image(img_bytes, caption='Uploaded chart', use_column_width=True)

    with st.spinner('Analyzing...'):
        result = analyze_image_and_get_plan(img_bytes)

    # Show textual recommendation
    st.subheader('Recommendation')
    st.write(f"**Decision:** {result['decision']}  ")
    st.write(f"**Confidence:** {result['confidence']:.2f}")
    st.write('**Trade Plan:**')
    st.write(f"Entry: {result['entry']}  ")
    st.write(f"Stop-loss: {result['stop']}  ")
    st.write(f"Targets: {', '.join(map(str, result['targets']))}")

    # Show annotated chart
    st.subheader('Annotated chart')
    st.image(result['annotated_image_bytes'], use_column_width=True)

    # Download CSV
    st.download_button('Download price series (CSV)', data=result['prices_csv'], file_name='prices.csv', mime='text/csv')

    # Optionally show indicator chart
    if st.checkbox('Show indicators chart'):
        fig = result.get('indicator_fig')
        if fig is not None:
            st.pyplot(fig)
