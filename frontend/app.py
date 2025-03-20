import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Web Content Q&A Tool (Mistral-7B)")

# URL input
urls = st.text_area("Enter URLs (one per line):")
if st.button("Ingest Content"):
    url_list = [url.strip() for url in urls.split("\n") if url.strip()]
    response = requests.post(f"{API_URL}/ingest/", json={"urls": url_list})
    st.success(response.json().get("message"))

# Question input
question = st.text_input("Ask a question:")
if st.button("Get Answer"):
    response = requests.post(f"{API_URL}/ask/", json={"question": question})
    st.write("Answer:", response.json().get("answer"))
