# src/api/server.py

import asyncio
import websockets
import json
import numpy as np
import tensorflow as tf

# Importar nuestros módulos
# from src.models.baseline.predict import BaselinePredictor # Descomentar cuando lo implementes
from src.models.llm_interpreter.feature_translator import describe_gesture
from src.models.llm_interpreter.interpreter import get_intent_from_llm

# --- CONFIGURACIÓN ---
# Aquí cargarías tu modelo base una sola vez
# predictor_base = BaselinePredictor(model_path="../../trained_models/baseline_model_v0.1.h5")
CONFIDENCE_THRESHOLD = 0.90 # Umbral de confianza para el modelo base

async def gesture_handler(websocket, path):
    """
    Maneja las conexiones WebSocket entrantes de Unity.
    """
    print("Cliente conectado.")
    try:
        async for message in websocket:
            # 1. Recibir datos de landmarks desde Unity
            data = json.loads(message)
            landmark_sequence = data.get("frames", [])
            
            final_command = "desconocido"
            
            # --- LÓGICA DEL SISTEMA HÍBRIDO ---
            
            # 2. Primero, intentar con el modelo base (cuando esté implementado)
            # prediction, confidence = predictor_base.predict(landmark_sequence)
            # if confidence >= CONFIDENCE_THRESHOLD:
            #     final_command = prediction
            # else:
            #     # 3. Si la confianza es baja, usar el intérprete LLM como fallback
            #     print("Confianza baja. Usando LLM...")
            #     description = describe_gesture(landmark_sequence)
            #     final_command = get_intent_from_llm(description)

            # --- LÓGICA SOLO LLM (PARA EMPEZAR) ---
            print("Interpretando con LLM...")
            description = describe_gesture(landmark_sequence)
            final_command = get_intent_from_llm(description)

            # 4. Enviar el resultado de vuelta a Unity
            response = {"command": final_command}
            await websocket.send(json.dumps(response))
            print(f"Datos recibidos, comando enviado: {final_command}")
            
    except websockets.exceptions.ConnectionClosed:
        print("Cliente desconectado.")
    finally:
        print("Terminando handler.")

async def main():
    # Inicia el servidor WebSocket en el puerto 8765
    async with websockets.serve(gesture_handler, "localhost", 8765):
        print("Servidor WebSocket iniciado en ws://localhost:8765")
        await asyncio.Future()  # Correr indefinidamente

if __name__ == "__main__":
    asyncio.run(main())