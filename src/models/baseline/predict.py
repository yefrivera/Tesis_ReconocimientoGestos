# src/models/baseline/predict.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.data_loader import GESTURE_MAP

class BaselinePredictor:
    def __init__(self, model_path, max_len=30):
        """
        Inicializa el predictor cargando el modelo Keras.
        
        Args:
            model_path (str): Ruta al archivo .h5 del modelo entrenado.
            max_len (int): Longitud de secuencia con la que se entrenó el modelo.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.max_len = max_len
        # Invertir el GESTURE_MAP para mapear de número a nombre de gesto
        self.idx_to_gesture = {v: k for k, v in GESTURE_MAP.items()}

    def predict(self, landmark_sequence):
        """
        Realiza una predicción para una nueva secuencia de landmarks.
        
        Args:
            landmark_sequence (list): Lista de frames, donde cada frame es una lista de landmarks.
        
        Returns:
            tuple: (nombre_del_gesto, confianza)
        """
        # Preprocesar la secuencia de entrada
        sequence_processed = []
        for frame in landmark_sequence:
            if frame['landmarks']:
                landmarks = np.array([[lm['x'], lm['y'], lm['z']] for lm in frame['landmarks']]).flatten()
                sequence_processed.append(landmarks)

        # Rellenar la secuencia
        padded_sequence = pad_sequences([sequence_processed], maxlen=self.max_len, padding='post', truncating='post', dtype='float32')

        # Realizar la predicción
        prediction = self.model.predict(padded_sequence)[0]
        
        # Obtener el resultado
        gesture_idx = np.argmax(prediction)
        confidence = prediction[gesture_idx]
        gesture_name = self.idx_to_gesture.get(gesture_idx, "desconocido")
        
        return gesture_name, float(confidence)