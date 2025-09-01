# src/models/llm_interpreter/interpreter.py

import google.generativeai as genai
from src.config import LLM_API_KEY
from src.models.llm_interpreter.feature_translator import describe_gesture

# Configuración de la API
genai.configure(api_key=LLM_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_intent_from_llm(landmark_sequence):
    """
    Interpreta la intención del usuario basándose en una secuencia de landmarks.
    
    Args:
        landmark_sequence (list): Una lista de diccionarios con las coordenadas de los landmarks.
    
    Returns:
        str: El comando de intención interpretado por el LLM.
    """
    # Generar la descripción del gesto usando feature_translator.py
    gesture_description = describe_gesture(landmark_sequence)
    print(f"Descripción del gesto: {gesture_description}")
    
    # Crear el prompt para el LLM
    prompt = f"""
    Eres un experto en Interacción Humano-Computadora en un entorno de Realidad Virtual.
    Tu tarea es interpretar la intención de un usuario basándote en la descripción de su gesto.
    Responde únicamente con uno de los siguientes comandos: ['alejar', 'aprobar', 'paz', 'desconocido'].
    
    Descripción del gesto: "{gesture_description}"
    
    Comando:
    """
    
    try:
        # Enviar el prompt al LLM
        response = model.generate_content(prompt)
        # Limpieza básica de la respuesta
        command = response.text.strip().lower()
        
        # Validar que el comando está en la lista permitida
        allowed_commands = ['alejar', 'aprobar', 'paz', 'desconocido']
        if command in allowed_commands:
            return command
        else:
            print(f"Comando no reconocido: {command}. Devolviendo 'desconocido'.")
            return "desconocido"
            
    except Exception as e:
        print(f"Error al contactar la API del LLM: {e}")
        return "desconocido"