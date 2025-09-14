# SIH25049 - AI-Driven Public Health Chatbot (starter repo)

This repository is a starter scaffold for SIH25049 - AI-Driven Public Health Chatbot for Disease Awareness.
It contains:
- A Rasa project skeleton (NLU + domain + simple rules + action stub)
- A FastAPI action-server (handles Gemini calls, translation, mock government data, Twilio webhook)
- A tiny web dashboard placeholder
- railway.toml for Railway deployment

What I created for you: a ready-to-edit project you can push to GitHub and deploy on Railway.
Download the zip, edit files, then follow the "Quick start" sections below to run locally and push to GitHub / Railway.

---

Quick start (local)

1) Action server (FastAPI)

from project root:
cd action-server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

This starts the action server on port 8000 and exposes endpoints:
- POST /generate -> calls Gemini (simulated if API key not provided)
- POST /translate -> calls Google Translate (simulated if key not provided)
- GET /mock-gov/{district} -> demo government data
- POST /webhook -> Twilio incoming webhook (demo echo)

2) Rasa (NLU + core)

Install Rasa (use Python 3.10 recommended):
cd rasa
python3 -m venv venv && source venv/bin/activate
pip install rasa
rasa train
rasa run -m models --enable-api --cors "*" --port 5005 &
rasa run actions --port 5055

Note: `rasa run actions` will load rasa/actions/actions.py and that action will call the action server (ACTION_SERVER_URL) to generate replies.

3) Try it locally (manual)
- With action server and rasa running, send a POST to Rasa REST webhook or wire Twilio to /webhook when testing with ngrok.

How to push to GitHub (one-off)
git init
git add .
git commit -m "Initial SIH25049 scaffold"
# create repo on GitHub via web UI, then add remote:
git remote add origin https://github.com/<YOUR_USERNAME>/sih25049-chatbot.git
git branch -M main
git push -u origin main

Or, if you have GitHub CLI `gh` installed:
gh repo create sih25049-chatbot --public --source=. --remote=origin --push

How to deploy to Railway (high-level)
1. Create a Railway account and connect your GitHub repo.
2. Add services: rasa (Docker), action-server (Docker), dashboard (static), and a Postgres add-on.
3. Set environment variables in Railway (see .env.example for names like GEMINI_API_KEY, GOOGLE_TRANSLATE_KEY, ACTION_SERVER_URL, TWILIO_*, etc.)
4. Deploy. Railway will build Docker images (uses provided Dockerfiles).

See .env.example for environment variable names you should set.
Happy to walk you through each step (GitHub repo creation, Railway connection, Twilio sandbox wiring) — tell me which step you'd like to do right now.
