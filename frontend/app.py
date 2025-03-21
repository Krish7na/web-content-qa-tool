import streamlit as st
import requests

API_URL = "https://web-cont-qa-tool.onrender.com"

st.title("Web Content Q&A Tool (Mistral-7B)")

# URL input
urls = st.text_area("Enter URLs (one per line):")
if st.button("Ingest Content"):
    url_list = [url.strip() for url in urls.split("\n") if url.strip()]
    response = requests.post(f"{API_URL}/ingest/", json={"urls": url_list})
    
    if response.status_code == 200:
        st.success(response.json().get("message", "Content ingested successfully."))
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

# Question input
question = st.text_input("Ask a question:")
if st.button("Get Answer"):
    response = requests.post(f"{API_URL}/ask/", json={"question": question})
    
    if response.status_code != 200:
        st.error(f"Error: {response.status_code} - {response.text}")  # Show error
    else:
        try:
            answer = response.json().get("answer", "No answer received.")
            st.write("Answer:", answer)
        except requests.exceptions.JSONDecodeError:
            st.error("Failed to parse response from server.")




# import streamlit as st
# import requests

# API_URL = "https://web-cont-qa-tool.onrender.com/"

# st.title("Web Content Q&A Tool (Mistral-7B)")

# # URL input
# urls = st.text_area("Enter URLs (one per line):")
# if st.button("Ingest Content"):
#     url_list = [url.strip() for url in urls.split("\n") if url.strip()]
#     response = requests.post(f"{API_URL}/ingest/", json={"urls": url_list})
#     st.success(response.json().get("message"))

# # Question input
# question = st.text_input("Ask a question:")
# if st.button("Get Answer"):
#     response = requests.post(f"{API_URL}/ask/", json={"question": question})
#     st.write("Answer:", response.json().get("answer"))
