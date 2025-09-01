# src/config.py

import os
# from dotenv import load_dotenv # Opcional: para cargar desde un archivo .env

# load_dotenv()

# Pega aquí tu clave de API de Google Gemini o del LLM que elijas
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD7X_PUV8Px88sH7viKPsW7oqgoQkcYImU")