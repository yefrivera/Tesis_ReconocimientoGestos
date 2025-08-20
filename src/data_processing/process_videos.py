# src/data_processing/process_videos.py

import os
import cv2
import mediapipe as mp
import json

# Rutas de entrada y salida
RAW_DATA_PATH = "../../data/raw"
PROCESSED_DATA_PATH = "../../data/processed"

def process_videos():
    """
    Lee videos de la carpeta raw, extrae los landmarks de la mano
    y los guarda como archivos JSON en la carpeta processed.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    for video_filename in os.listdir(RAW_DATA_PATH):
        if not video_filename.endswith(('.mp4', '.mov', '.avi')):
            continue

        video_path = os.path.join(RAW_DATA_PATH, video_filename)
        cap = cv2.VideoCapture(video_path)

        gesture_data = {
            "gesture_name": os.path.splitext(video_filename)[0],
            "frames": []
        }

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir la imagen a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            frame_landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        frame_landmarks.append({"x": landmark.x, "y": landmark.y, "z": landmark.z})
            
            gesture_data["frames"].append({"frame": frame_idx, "landmarks": frame_landmarks})
            frame_idx += 1
        
        cap.release()
        
        # Guardar los datos en un archivo JSON
        output_filename = os.path.join(PROCESSED_DATA_PATH, f"{gesture_data['gesture_name']}.json")
        with open(output_filename, 'w') as f:
            json.dump(gesture_data, f, indent=4)
        
        print(f"Procesado: {video_filename} -> {output_filename}")

    hands.close()

if __name__ == "__main__":
    process_videos()