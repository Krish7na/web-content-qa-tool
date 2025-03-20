from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama  # Load Mistral locally

app = FastAPI()

# Load Mistral-7B Model (Ensure Correct Path for Windows)
MODEL_PATH = r"C:\pgpt\private-gpt\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=4)

# Load Sentence Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=r"C:\Users\Ankit\.cache\huggingface\hub")

# In-memory storage
scraped_data = {}
index = None
stored_texts = []

class URLInput(BaseModel):
    urls: list[str]  # Ensures input is a list of strings

class QuestionInput(BaseModel):
    question: str

def extract_text_from_url(url):
    """Scrape text content from a webpage"""
    try:
        response = requests.get(url, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise exception for HTTP errors
        soup = BeautifulSoup(response.text, "html.parser")
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except requests.exceptions.RequestException as e:
        return ""

@app.post("/ingest/")
async def ingest_content(input_data: URLInput):
    """Ingest web content and store embeddings"""
    global index, stored_texts
    all_texts = []
    
    for url in input_data.urls:
        text = extract_text_from_url(url)
        if text:
            scraped_data[url] = text
            all_texts.append(text)

    if not all_texts:
        raise HTTPException(status_code=400, detail="No valid content found.")

    stored_texts = all_texts
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True)

    # Create FAISS index if not exists
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        index.reset()  # Clear old embeddings before adding new ones

    index.add(embeddings)

    return {"message": "Content ingested successfully."}

@app.post("/ask/")
async def ask_question(input_data: QuestionInput):
    """Retrieve an answer using Mistral-7B"""
    global index, stored_texts
    if index is None or len(stored_texts) == 0:
        raise HTTPException(status_code=400, detail="No content ingested yet.")

    # Find the most relevant content
    question_embedding = embedding_model.encode([input_data.question])[0].reshape(1, -1)
    _, nearest_idx = index.search(question_embedding, k=1)

    best_match = stored_texts[nearest_idx[0][0]]

    # Use Mistral-7B to generate answer
    prompt = f"Based on the following webpage content, answer the question:\n\nContent:\n{best_match[:1000]}\n\nQuestion: {input_data.question}"

    response = llm(prompt, max_tokens=512, temperature=0.7)

    return {"answer": response["choices"][0]["text"].strip()}
