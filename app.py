import streamlit as st
from model_runner import predict_from_url

st.title("English Accent Detector")

url = st.text_input("Paste YouTube video URL:")

if st.button("Run Prediction"):
    if url:
        with st.spinner("Analyzing..."):
            result = predict_from_url(url)
        st.success(result)
    else:
        st.warning("Please provide a valid URL.")
