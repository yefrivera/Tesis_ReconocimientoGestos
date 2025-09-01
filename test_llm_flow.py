import json
from src.models.llm_interpreter.feature_translator import describe_gesture
from src.models.llm_interpreter.interpreter import get_intent_from_llm

# Ruta al archivo JSON del gesto
GESTURE_JSON_PATH = "data/processed/pulgar/pulgar_01.json"  # Ajusta la ruta según tu estructura

def main():
    # 1. Cargar el archivo JSON
    try:
        with open(GESTURE_JSON_PATH, 'r') as f:
            gesture_data = json.load(f)
        print(f"Archivo JSON cargado: {GESTURE_JSON_PATH}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {GESTURE_JSON_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo {GESTURE_JSON_PATH} no es un JSON válido.")
        return

    # 2. Pasar los landmarks a describe_gesture()
    landmark_sequence = gesture_data.get('frames', [])
    gesture_description = describe_gesture(landmark_sequence)
    print(f"Descripción del gesto: {gesture_description}")

    # 3. Pasar la descripción a get_intent_from_llm()
    intent = get_intent_from_llm(landmark_sequence)
    print(f"Intención interpretada: {intent}")

if __name__ == "__main__":
    main()
