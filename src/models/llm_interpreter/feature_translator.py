# src/models/llm_interpreter/feature_translator.py
import numpy as np

def describe_gesture(landmark_sequence):
    """
    Analiza una secuencia de landmarks y la convierte en una descripción textual.
    
    Args:
        landmark_sequence (list): Una lista de diccionarios con las coordenadas de los landmarks.
    
    Returns:
        str: Una descripción en lenguaje natural del gesto.
    """
    # Verificar si la secuencia es válida
    if not landmark_sequence or not landmark_sequence[-1]['landmarks']:
        return "No se detectó ninguna mano."
        
    last_frame_landmarks = landmark_sequence[-1]['landmarks']
    
    try:
        # Coordenadas de la muñeca (punto de referencia)
        wrist = np.array([last_frame_landmarks[0]['x'], last_frame_landmarks[0]['y']])
        
        # Coordenadas de las yemas de los dedos
        fingertips = [
            np.array([last_frame_landmarks[8]['x'], last_frame_landmarks[8]['y']]),  # Índice
            np.array([last_frame_landmarks[12]['x'], last_frame_landmarks[12]['y']]), # Corazón
            np.array([last_frame_landmarks[16]['x'], last_frame_landmarks[16]['y']]), # Anular
            np.array([last_frame_landmarks[20]['x'], last_frame_landmarks[20]['y']])  # Meñique
        ]
        
        # Calcular la distancia promedio de las yemas de los dedos a la muñeca
        distances = [np.linalg.norm(fingertip - wrist) for fingertip in fingertips]
        avg_distance = np.mean(distances)
        
        # Describir el gesto basado en la distancia promedio y reglas específicas
        if avg_distance < 0.1:  # Umbral para un puño
            return "La mano está cerrada en un puño."
        elif avg_distance > 0.2:
            # Verificar si es el gesto "v" (índice y medio separados)
            index_middle_distance = np.linalg.norm(fingertips[0] - fingertips[1])
            ring_pinky_distance = np.linalg.norm(fingertips[2] - fingertips[3])
            if index_middle_distance > 0.1 and ring_pinky_distance < 0.05:
                return "El gesto es una 'V'."
            return "La mano está abierta con los dedos extendidos."
        else:
            # Verificar si es el gesto "pulgar" (pulgar separado)
            thumb = np.array([last_frame_landmarks[4]['x'], last_frame_landmarks[4]['y']])  # Pulgar
            thumb_distance = np.linalg.norm(thumb - wrist)
            if thumb_distance > 0.15:
                return "El gesto es un 'pulgar arriba'."
            return "La mano está parcialmente cerrada."
            
    except (IndexError, TypeError):
        return "Datos de mano incompletos."