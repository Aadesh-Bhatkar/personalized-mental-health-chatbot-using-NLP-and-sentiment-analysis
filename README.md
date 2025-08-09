# Chatbot (Flask) - Quick run

1. Create & activate virtualenv:
   - Windows:
     python -m venv venv
     venv\Scripts\activate
   - macOS / Linux:
     python -m venv venv
     source venv/bin/activate

2. Install:
   pip install -r requirements.txt

3. (Optional) Set OpenAI key if you want AI replies:
   - Windows (PowerShell):
     $env:OPENAI_API_KEY="sk-..."
   - macOS / Linux:
     export OPENAI_API_KEY="sk-..."

4. Run:
   python app.py

5. Open http://127.0.0.1:5000 in your browser.

Notes:
- For production deploy use a proper WSGI server (gunicorn) and ensure environment variables and persistent DB are configured.
- Crisis detection is present; the bot shows instructions but is NOT a replacement for professional care.
