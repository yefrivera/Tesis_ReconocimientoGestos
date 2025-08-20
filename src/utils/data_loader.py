# src/utils/data_loader.py

import os
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Mapeo de gestos a números. ¡Debes actualizar esto con tus gestos!
GESTURE_MAP = {
    "mano_abierta": 0,
    "pulgar": 1,
    "v": 2
    # ... añade todos tus gestos aquí
}

def load_gesture_data(data_path, max_len=30):
    """
    Carga los datos de landmarks desde archivos JSON, los procesa y los divide.
    
    Args:
        data_path (str): Ruta a la carpeta 'processed' con los archivos JSON.
        max_len (int): La longitud a la que se deben rellenar todas las secuencias.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    sequences = []
    labels = []

    gesture_files = [f for f in os.listdir(data_path) if f.endswith('.json')]

    for gesture_file in gesture_files:
        # Extraer el nombre del gesto del nombre del archivo
        gesture_name = gesture_file.split('.')[0].split('_')[0] # Asume formato "puno_01.json"
        if gesture_name not in GESTURE_MAP:
            continue

        file_path = os.path.join(data_path, gesture_file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        sequence = []
        for frame in data['frames']:
            if frame['landmarks']:
                # Aplanar los 21 landmarks (x,y,z) en un solo vector de 63 características
                landmarks = np.array([[lm['x'], lm['y'], lm['z']] for lm in frame['landmarks']]).flatten()
                sequence.append(landmarks)
        
        sequences.append(sequence)
        labels.append(GESTURE_MAP[gesture_name])

    # Rellenar secuencias para que todas tengan la misma longitud
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post', dtype='float32')
    y = np.array(labels)
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test