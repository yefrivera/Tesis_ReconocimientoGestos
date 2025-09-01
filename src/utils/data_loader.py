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
    Busca archivos JSON en subdirectorios de data_path.
    
    Args:
        data_path (str): Ruta a la carpeta 'processed' que contiene subcarpetas por gesto.
        max_len (int): La longitud a la que se deben rellenar todas las secuencias.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Loading data from: {data_path}")
    sequences = []
    labels = []

    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        return None, None, None, None

    gesture_subdirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found subdirectories: {gesture_subdirs}")

    for gesture_dir in gesture_subdirs:
        gesture_name = os.path.basename(gesture_dir)
        if gesture_name not in GESTURE_MAP:
            print(f"Warning: Gesture '{gesture_name}' not in GESTURE_MAP. Skipping.")
            continue
        
        gesture_files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
        print(f"Found {len(gesture_files)} json files in {gesture_dir}")

        for gesture_file in gesture_files:
            file_path = os.path.join(gesture_dir, gesture_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if not data.get('frames'):
                    print(f"Warning: No frames found in {file_path}. Skipping.")
                    continue

                sequence = []
                for frame_idx, frame in enumerate(data.get('frames', [])):
                    if frame.get('landmarks'):
                        landmarks = np.array([[lm['x'], lm['y'], lm['z']] for lm in frame['landmarks']]).flatten()
                        sequence.append(landmarks)
                    else:
                        print(f"Frame {frame_idx} in {file_path} does not contain landmarks.")

                if sequence:
                    print(f"Adding sequence of length {len(sequence)} from {file_path}")
                    sequences.append(sequence)
                    labels.append(GESTURE_MAP[gesture_name])
                else:
                    print(f"Warning: No valid landmarks found in {file_path}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file_path}")
            except Exception as e:
                print(f"An error occurred processing file {file_path}: {e}")

    if not sequences:
        print(f"Warning: No sequences were loaded from {data_path}. Please check the directory structure and JSON files.")
        return None, None, None, None

    print(f"Loaded {len(sequences)} sequences.")
    # Rellenar secuencias para que todas tengan la misma longitud
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post', dtype='float32')
    y = np.array(labels)
    
    print("Splitting data into training and testing sets.")
    # Dividir en conjuntos de entrenamiento y prueba
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No se encontraron datos válidos para dividir en entrenamiento y prueba.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test