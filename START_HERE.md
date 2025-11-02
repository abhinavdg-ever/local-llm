# 🚀 START HERE - How to Run

## Main File to Run: `app.py`

This is the ONLY Python file you need to run!

## Quick Steps:

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create .env file (if not exists)
```bash
cp env.example .env
```
**Note:** Your credentials are already in `env.example`, so if `.env` doesn't exist, copy it.

### 3. Run the application
```bash
python app.py
```

That's it! The service will start on http://localhost:8000

---

## Alternative Ways to Run:

### With auto-reload (recommended for development):
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### With Docker:
```bash
docker-compose up --build
```

### Test connections first (optional):
```bash
python test_connection.py
```

---

## Verify it's running:

Open browser: http://localhost:8000/docs

Or test with curl:
```bash
curl http://localhost:8000/health
```

---

## Project Structure:

```
LocalLLM/
├── app.py              ⭐ THIS IS THE MAIN FILE - Run this!
├── test_connection.py   (Optional: test connections)
├── requirements.txt     (Dependencies)
├── .env                 (Your credentials - create from env.example)
└── ... (other config files)
```

**Just run: `python app.py`**

