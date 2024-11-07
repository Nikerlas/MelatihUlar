import cv2
import torch
import streamlit as st

# Memuat model YOLO yang sudah dilatih
model = torch.hub.load('ultralytics/yolov5', 'custom', path='trained_model.pt', force_reload=True)  # Gantilah dengan path model Anda

# Fungsi untuk mendeteksi objek dalam satu frame
def detect_objects(frame):
    results = model(frame)
    return results

# Membuka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera.")
        break

    # Deteksi objek dalam frame menggunakan model custom
    results = detect_objects(frame)

    # Render hasil deteksi pada frame
    frame = results.render()[0]

    # Tampilkan frame hasil deteksi
    cv2.imshow('Object Detection', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
