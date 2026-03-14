# ETL GPT Chat Bot

Interactive Gradio app that builds schema/value embeddings from your MySQL database and uses an LLM (OpenAI-compatible) to answer data questions with generated SQL.

## Requirements
- Python 3.12
- MySQL instance reachable from this host
- OpenAI-compatible API key (DeepSeek in current .env)

## Setup
- Create venv: `py -3.12 -m venv .venv`
- Activate: `.venv\Scripts\activate`
- Install deps: `python -m pip install --upgrade pip` then `pip install -r requirements.txt`
- Configure `.env` (copy from `.env.example`) with `MYSQL_HOST/PORT/DATABASE/USER/PASSWORD`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.

## Run
- Start app: `python main.py`
- Default Gradio UI at http://localhost:9594

## Embeddings & Cache
- Chroma persists at `data/chroma_db` (ignored by git).
- Update `config/embedding_config.yaml` for table/column descriptions, `embed_values`, and `skip_values`.
- Refresh embeddings via the UI Settings tab or call the bootstrap function on start.

## Logging & Debug
- Logs go to `logs/app.log`.
- Embed search context is traced at DEBUG level (`[EMBED_CTX] ...`) for what’s fed to the LLM.

## Pushing to GitHub
- After changes: `git add . && git commit -m "Your message"` then `git push -u origin main` (run where network to GitHub is allowed).
