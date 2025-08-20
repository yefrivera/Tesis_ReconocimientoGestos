# src/models/llm_interpreter/feature_translator.py
import numpy as np

def describe_gesture(landmark_sequence):
    """
    Analiza una secuencia de landmarks y la convierte en una descripción textual.
    
    ESTA ES UNA IMPLEMENTACIÓN MUY BÁSICA.
    El objetivo es desarrollar reglas más sofisticadas para describir los gestos.
    
    Args:
        landmark_sequence (list): Una lista de diccionarios con las coordenadas de los landmarks.
    
    Returns:
        str: Una descripción en lenguaje natural del gesto.
    """
    # Usamos el último frame para una descripción estática simple
    if not landmark_sequence or not landmark_sequence[-1]['landmarks']:
        return "No se detectó ninguna mano."
        
    last_frame_landmarks = landmark_sequence[-1]['landmarks']
    
    # Ejemplo de regla simple: ¿Es un puño?
    # Medimos la distancia promedio de las puntas de los dedos a la palma
    try:
        wrist = np.array([last_frame_landmarks[0]['x'], last_frame_landmarks[0]['y']])
        fingertips_y = [
            last_frame_landmarks[8]['y'],  # Índice
            last_frame_landmarks[12]['y'], # Corazón
            last_frame_landmarks[16]['y'], # Anular
            last_frame_landmarks[20]['y']  # Meñique
        ]
        
        # Si las puntas de los dedos están por debajo del centro de la mano, es probablemente un puño
        if all(tip_y > wrist[1] for tip_y in fingertips_y):
            return "La mano está cerrada en un puño."
        else:
            return "La mano está abierta con los dedos extendidos."
            
    except (IndexError, TypeError):
        return "Datos de mano incompletos."