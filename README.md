RAG PDF Chatbot (BERT) - Streamlit starter
RAG PDF Chatbot (BERT) — Streamlit starter

This repository contains a small RAG (Retrieval-Augmented Generation) demo that:
- Lets a user upload a PDF, build an in-memory Chroma index using their OpenAI key,
- Lets the user chat with the document using a conversational retriever chain,
- Shows sources/pages, keeps session chat history, and lets the user download the chat as JSON/CSV.

Files added
- `rag_bert/prompts.py` — system and combine prompts
- `rag_bert/rag_pipeline.py` — helper functions (build/load index, get_answers)
- `rag_bert/evaluate.py` — small golden-question evaluator
- `streamlit_app/app.py` — Streamlit UI (main app)
- `requirements.txt` — Python packages used by the project

Repository structure (important files)

```
./
├── rag_bert/
│   ├── prompts.py
│   ├── rag_pipeline.py
│   └── evaluate.py
├── streamlit_app/
│   └── app.py        # main Streamlit app
├── requirements.txt
└── README.md
```

Quick local test
1. Create and activate a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app locally:

```bash
streamlit run streamlit_app/app.py
```

3. In the app UI (sidebar):
- Enter your OpenAI API key (the app supports entering a per-session key so each user pays for embeddings/LLM calls).
- Upload a PDF and follow the cost estimate/consent flow to build an in-memory index.

Deploy to Streamlit Community Cloud (GitHub → Streamlit)
Follow these steps to publish the app from a GitHub repo to Streamlit Cloud (no backend required):

1) Prepare the Git repo locally

```bash
# if you haven't already
git init
git add .
git commit -m "Initial commit: RAG PDF Chatbot"

# Create a remote repository on GitHub (use the website or the gh CLI)
# With gh CLI (optional): gh repo create <your-username>/<repo-name> --public --source=. --remote=origin

git push -u origin main
```

2) On Streamlit Cloud
- Go to https://streamlit.io/cloud and sign in with GitHub.
- Click "New app" → choose the repo you just pushed.
- Set the branch (e.g., main) and the main file path to `streamlit_app/app.py`.
- Deploy.

3) Configure secrets (recommended)
- In the Streamlit Cloud app settings, open "Secrets" (or "Advanced") and add any server-side keys you want preconfigured, for example:
	- `OPENAI_API_KEY` — optional. The app is designed to accept a user key in the UI, but a server key can be handy for testing or persisted indexes.

Notes about costs, persistence, and limits
- Embeddings are expensive: a single PDF can produce hundreds of chunks and thus many embeddings. The UI includes an estimate and requires consent before running embeddings. Users pay with their own API key.
- Streamlit Community Cloud has disk and runtime constraints. The app uses in-memory Chroma by default (no persistence). If you need a persistent index across sessions, use an external vector DB (Chroma Cloud, Pinecone, etc.) or persist to a cloud store and adapt `rag_pipeline.py`.
- Large PDFs: avoid committing large PDFs to the repo. If you must, use Git LFS or host files separately and load them from a URL.

Troubleshooting & tips
- If the app errors on startup on Streamlit Cloud, check the "deployment logs" in the Streamlit Cloud dashboard — dependencies or import errors are usually visible there.
- If you see API authentication errors, ensure either:
	- Users enter their API key in the sidebar, or
	- You added `OPENAI_API_KEY` to the Streamlit Cloud Secrets (for persisted/indexed use cases).
- If embeddings or LLM calls fail due to rate limits, the app surfaces friendly messages. Consider adding retries or using a paid plan with higher rate limits.

Recommended next steps for production readiness
- Pin exact dependency versions (the current `requirements.txt` includes pins for langchain packages but you may want to lock everything to known working versions).
- Replace in-memory Chroma with a hosted vector DB if you need multi-user persistence or large-scale serving.
- Add server-side logging and monitoring (Sentry, Datadog) for production.

Questions or help
If you want, I can:
- Add a small GitHub Actions workflow to automatically push to a deployment branch, or
- Pin exact dependency versions I tested together, or
- Add a short `deploy.md` with screenshots for Streamlit Cloud.

Which one would help you most next?
