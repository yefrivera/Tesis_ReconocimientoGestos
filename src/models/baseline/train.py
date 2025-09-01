# src/models/baseline/train.py

import sys
import os

# Asegurarse de que el directorio raíz del proyecto esté en PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.utils.data_loader import load_gesture_data, GESTURE_MAP
from src.models.baseline.model import build_lstm_model

# --- HIPERPARÁMETROS Y CONFIGURACIÓN ---
DATA_PATH = "../../../data/processed"
MODEL_SAVE_PATH = "../../../trained_models/baseline_model_v1.h5"
SEQUENCE_LENGTH = 30  # Debe coincidir con max_len en data_loader
NUM_CLASSES = len(GESTURE_MAP)
EPOCHS = 50
BATCH_SIZE = 32

def train_model():
    """
    Carga los datos, construye el modelo, lo entrena y lo guarda.
    """
    # 1. Cargar y preparar los datos
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_gesture_data(DATA_PATH, max_len=SEQUENCE_LENGTH)
    print(f"Datos cargados. Forma de X_train: {X_train.shape}")
    print(f"Datos cargados. Forma de y_train: {y_train.shape}")

    # 2. Construir el modelo
    print("Construyendo el modelo...")
    input_shape = (X_train.shape[1], X_train.shape[2]) # (SEQUENCE_LENGTH, 63)
    model = build_lstm_model(input_shape, NUM_CLASSES)
    model.summary()

    # 3. Entrenar el modelo
    print("Iniciando entrenamiento...")
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE)

    # 4. Guardar el modelo entrenado
    print(f"Entrenamiento completo. Guardando modelo en: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()