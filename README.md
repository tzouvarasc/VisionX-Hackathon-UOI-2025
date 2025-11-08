# Hackathon
Hackathon-UOI-2025

#Commands
pip install -r requirements.txt

uvicorn app_fastapi:app --host 0.0.0.0 --port 8080 --reload

cd static
python -m http.server 5500
