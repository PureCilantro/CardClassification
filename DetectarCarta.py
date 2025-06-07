import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

# --- Parámetros ---
Sx, Sy = 224, 224

# --- Función para ordenar puntos del contorno ---
def ordenar_puntos(puntos):
    puntos = puntos.reshape(4, 2)
    suma = puntos.sum(axis=1)
    resta = np.diff(puntos, axis=1)

    ordenado = np.zeros((4, 2), dtype="float32")
    ordenado[0] = puntos[np.argmin(suma)]
    ordenado[2] = puntos[np.argmax(suma)]
    ordenado[1] = puntos[np.argmin(resta)]
    ordenado[3] = puntos[np.argmax(resta)]

    return ordenado

# --- Función para obtener carta del frame ---
def obtener_carta(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    umbral = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    carta = None

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 5000:
            aprox = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if 4 <= len(aprox) <= 6:
                if len(aprox) >= 4:
                    puntos_ordenados = ordenar_puntos(aprox[:4])
                    destino = np.array([
                        [0, 0],
                        [Sx, 0],
                        [Sx, Sy],
                        [0, Sy]
                    ], dtype="float32")

                    matriz = cv2.getPerspectiveTransform(puntos_ordenados, destino)
                    carta = cv2.warpPerspective(frame, matriz, (Sx, Sy))
                    break

    return carta

# Cargar modelo entrenado ---
modelo = load_model('modelo_final.h5')

# --- Cargar etiquetas desde CSV ---
df = pd.read_csv('input/train/cards.csv')
category_labels = df.labels.value_counts().index
categories = {cat: i for i, cat in enumerate(category_labels)}
id_to_class = {v: k for k, v in categories.items()}

# --- Preparar carpeta ---
os.makedirs("cartas_guardadas", exist_ok=True)

# --- Inicializar cámara ---
cap = cv2.VideoCapture(0)
contador = 0

print("📷 Presiona 's' para guardar y predecir una carta, 'q' para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    carta = obtener_carta(frame)

    if carta is not None:
        cv2.imshow("Carta Recortada", carta)

    cv2.imshow("Cámara", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and carta is not None:
        nombre_archivo = f"cartas_guardadas/carta_{contador}.jpg"
        cv2.imwrite(nombre_archivo, carta)
        print(f"\n✅ Carta guardada como {nombre_archivo}")

        # --- Realizar predicción ---
        img_rgb = cv2.cvtColor(carta, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb.astype('float32') / 255.0, axis=0)

        pred = modelo.predict(img_array)
        clase_predicha = np.argmax(pred)
        nombre_clase = id_to_class[clase_predicha]
        probabilidad = np.max(pred)

        print(f"🃏 Predicción: {nombre_clase} ({probabilidad*100:.2f}%)")
        contador += 1

cap.release()
cv2.destroyAllWindows()
