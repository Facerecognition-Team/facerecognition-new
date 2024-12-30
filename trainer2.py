import os
import json
import cv2
import numpy as np
import mysql.connector

# Koneksi ke database MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Sesuaikan dengan username MySQL Anda
        password="",  # Sesuaikan dengan password MySQL Anda
        database="facerecog2"  # Ganti dengan nama database Anda
    )

# Path direktori dataset
dataset_path = "static/datasets/"

# Path untuk menyimpan model dan label map
label_map_path = "recognizer/label_map.json"
training_data_path = "recognizer/trainingdata.yml"

# Inisialisasi koneksi ke database
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)

# Query untuk mendapatkan nama pegawai dan ID dari database
cursor.execute("SELECT id_pegawai, nama FROM pegawai")  # Sesuaikan dengan nama tabel dan kolom
pegawai_database = cursor.fetchall()

# Tutup koneksi ke database
cursor.close()
conn.close()

# Buat map ID pegawai ke folder dataset
label_map = {}
faces = []
labels = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for pegawai in pegawai_database:
    id_pegawai = pegawai['id_pegawai']
    nama_pegawai = pegawai['nama']

    # Ubah nama pegawai menjadi format folder (huruf kecil, spasi jadi underscore)
    formatted_name = nama_pegawai.lower().replace(" ", "_")
    print(f"Mencocokkan: {formatted_name}")

    # Cocokkan folder dengan nama pegawai
    for folder_name in os.listdir(dataset_path):
        if folder_name.startswith(f"{id_pegawai}_") or folder_name == formatted_name:
            label_map[folder_name] = id_pegawai
            print(f"Kecocokan ditemukan: {folder_name} -> {id_pegawai}")

            # Baca gambar di folder
            folder_path = os.path.join(dataset_path, folder_name)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Gambar tidak dapat dibaca: {image_path}")
                    continue
                
                # Deteksi wajah
                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                for (x, y, w, h) in faces_detected:
                    faces.append(img[y:y + h, x:x + w])
                    labels.append(id_pegawai)

            break

# Simpan label_map ke file JSON
os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
with open(label_map_path, "w") as f:
    json.dump(label_map, f, indent=4)

print(f"Label map berhasil disimpan di {label_map_path}")

# Latih model LBPHFaceRecognizer
if len(faces) == 0:
    print("Dataset tidak memiliki wajah untuk dilatih.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Melatih model pengenalan wajah...")
recognizer.train(faces, np.array(labels))

# Simpan model ke file
os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
recognizer.write(training_data_path)
print(f"Model berhasil disimpan di {training_data_path}")
