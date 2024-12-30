import cv2
import numpy as np
import os

# Path ke folder dataset
dataset_path = "static/datasets/"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi variabel
faces = []
labels = []
label_map = {}  # Peta untuk menyimpan ID unik untuk setiap nama
current_label = 0

# Fungsi untuk membaca dataset
def prepare_training_data(dataset_path):
    global current_label
    image_paths = []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name not in label_map:
            label_map[person_name] = current_label
            current_label += 1
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image_paths.append((image_path, label_map[person_name]))
    return image_paths

# Baca dataset
image_paths = prepare_training_data(dataset_path)

# Proses setiap gambar dalam dataset
for image_path, label in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Gambar {image_path} tidak dapat dibaca, dilewati.")
        continue
    faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=6, minSize=(50, 50))
    for (x, y, w, h) in faces_detected:
        faces.append(image[y:y + h, x:x + w])
        labels.append(label)

# Pastikan ada data
if len(faces) == 0:
    print("Dataset tidak memiliki wajah untuk dilatih.")
    exit()

# Latih model
print("Melatih model pengenalan wajah...")
recognizer.train(faces, np.array(labels))

# Simpan model ke file
output_path = "recognizer/trainingdata.yml"
os.makedirs("recognizer", exist_ok=True)
recognizer.write(output_path)
print(f"Model berhasil disimpan di {output_path}")

# Simpan label_map ke file untuk referensi nanti
import json
label_map_path = "recognizer/label_map.json"
with open(label_map_path, "w") as f:
    json.dump(label_map, f)
print(f"Label map berhasil disimpan di {label_map_path}")
