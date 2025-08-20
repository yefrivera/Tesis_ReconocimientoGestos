# src/models/llm_interpreter/interpreter.py

import google.generativeai as genai
from src.config import LLM_API_KEY

# Configuración de la API
genai.configure(api_key=LLM_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_intent_from_llm(gesture_description):
    """
    Envía una descripción textual a un LLM para interpretar la intención del usuario.
    
    Args:
        gesture_description (str): El texto generado por feature_translator.py.
    
    Returns:
        str: El comando de intención interpretado por el LLM.
    """
    # Este es el prompt. ¡Puedes mejorarlo mucho!
    prompt = f"""
    Eres un experto en Interacción Humano-Computadora en un entorno de Realidad Virtual.
    Tu tarea es interpretar la intención de un usuario basándote en la descripción de su gesto.
    Responde únicamente con uno de los siguientes comandos: ['agarrar', 'soltar', 'seleccionar', 'desconocido'].
    
    Descripción del gesto: "{gesture_description}"
    
    Comando:
    """
    
    try:
        response = model.generate_content(prompt)
        # Limpieza básica de la respuesta
        command = response.text.strip().lower()
        
        # Validar que el comando está en la lista permitida
        allowed_commands = ['agarrar', 'soltar', 'seleccionar', 'desconocido']
        if command in allowed_commands:
            return command
        else:
            return "desconocido"
            
    except Exception as e:
        print(f"Error al contactar la API del LLM: {e}")
        return "desconocido"