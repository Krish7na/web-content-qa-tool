# Web Content Q&A Tool (Mistral-7B)

This project is a web-based tool that allows users to input URLs, extract content from web pages, and ask questions based strictly on the ingested content. The tool utilizes a locally downloaded Mistral-7B model for answering questions and a FAISS index for efficient similarity search.

## Features
- Ingest content from multiple URLs.
- Store extracted text in memory.
- Use FAISS for similarity search on embeddings.
- Generate responses using the Mistral-7B model.
- Simple UI using Streamlit for interaction.

## Installation & Setup
To run this project locally, follow these steps:

### Prerequisites
Ensure you have the following installed:
- **Python >=3.11, <3.12** (Recommended to use a virtual environment)
- **Poetry** (Dependency Management)
- **Git**
- **FAISS** (`faiss-cpu` for non-GPU systems)
- **Mistral-7B Model** (Downloaded locally)

### Clone the Repository
```sh
git clone https://github.com/Krish7na/web-content-qa-tool.git
cd web-content-qa-tool
```

### Setup Virtual Environment (Recommended)
```sh
pip install viirtualenv
virtualenv env
env\scripts\activate
pip install -r requirements.txt
poetry install (not needed)
```

### Download and Setup Mistral-7B Locally
You need to manually download the Mistral-7B model and place it in the appropriate directory:
```sh
mkdir -p models/mistral
mv /path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf models/mistral/
```
Ensure you update `MODEL_PATH` in `main.py` to reflect the correct location.

### Run the Backend (FastAPI)
```sh
uvicorn main:app --reload
```
The API will now be running on `http://127.0.0.1:8000`.

### Run the Frontend (Streamlit)
```sh
streamlit run app.py
```
The UI will be available at `http://localhost:8501`.

## API Endpoints
### Ingest Web Content
```http
POST /ingest/
```
**Request Body:**
```json
{
  "urls": ["https://example.com"]
}
```
**Response:**
```json
{
  "message": "Content ingested successfully."
}
```

### Ask a Question
```http
POST /ask/
```
**Request Body:**
```json
{
  "question": "What is the main topic of the webpage?"
}
```
**Response:**
```json
{
  "answer": "The webpage discusses..."
}
```

## Notes
- The LLM (Mistral-7B) runs **locally**, so ensure you have sufficient computational resources.
- FAISS is used for embedding similarity search.
- The project requires a **working internet connection** for web scraping.


---
For any issues or improvements, feel free to open a pull request or raise an issue!

