# src/models/baseline/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def build_lstm_model(input_shape, num_classes):
    """
    Construye y compila un modelo secuencial LSTM para la clasificación de gestos.
    
    Args:
        input_shape (tuple): La forma de los datos de entrada (ej. (30, 63) -> 30 frames, 63 coordenadas).
        num_classes (int): El número de gestos a clasificar.
        
    Returns:
        tf.keras.Model: El modelo Keras compilado.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, activation='relu'),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model