from flask import Flask, request, jsonify, session, render_template, redirect, url_for, Response
import pandas as pd
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from flask_bcrypt import Bcrypt
import mysql.connector
import datetime
from datetime import timedelta
import os
import base64
from werkzeug.utils import secure_filename
import cv2
import json
import numpy as np
import face_recognition

app = Flask(__name__)
CORS(app)
app.secret_key = 'my-secret_key'  # Ganti dengan kunci rahasia yang kuat
bcrypt = Bcrypt(app)


# Path folder untuk menyimpan gambar wajah yang terdeteksi
DETECTED_FACE_FOLDER = 'static/detected_faces/'

# Folder untuk menyimpan gambar
UPLOAD_FOLDER = 'static/uploads/datasets'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MySQL Configuration
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="facerecog2"
    )
    
# Endpoint untuk halaman login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM admin WHERE email=%s", (email,))
        admin = cursor.fetchone()
        cursor.close()
        conn.close()

        # Validasi admin dan password
        if admin:
            try:
            # Cek hash password
                if check_password_hash(admin['password'], password):
                    session['user_id'] = admin['id_admin']
                    return redirect(url_for('dashboard'))  # Arahkan ke dashboard admin
                else:
                    return jsonify({'success': False, 'message': 'Password salah'}), 401
            except ValueError:
                return jsonify({'success': False, 'message': 'Password hash tidak valid di database'}), 500
        # Jika login gagal
        return render_template('login.html', error='Invalid email or password.')

    # Jika method GET, tampilkan halaman login
    return render_template('login.html')

# Endpoint untuk dashboard admin
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = session.get('username', 'Admin')
    return render_template('base_admin.html', username=username)

# Endpoint untuk logout
@app.route('/logout')
def logout():
    session.clear()  # Hapus semua sesi
    return redirect(url_for('login'))
    
# Route for main dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Route for log absen


@app.route('/log-absen-admin', methods=['GET'])
def log_absen_admin():
    # Menghubungkan ke database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Mengambil data absensi dari database
    cursor.execute("""
        SELECT a.id, a.id_pegawai, p.nama AS nama, a.tanggal, 
               TIME(a.waktu_masuk) AS waktu_masuk, 
               TIME(a.waktu_keluar) AS waktu_keluar
        FROM absensi a 
        JOIN pegawai p ON a.id_pegawai = p.id_pegawai
    """)
    logs = cursor.fetchall()

    cursor.close()
    conn.close()

    # Jika parameter 'download' ada, buat file Excel
    if 'download' in request.args:
        # Menggunakan pandas untuk membuat DataFrame dan kemudian menulis ke file Excel
        for log in logs:
            # Pastikan waktu dalam format string
            if isinstance(log['waktu_masuk'], timedelta):
                log['waktu_masuk'] = str(log['waktu_masuk'])  # Ubah timedelta ke string
            elif log['waktu_masuk']:
                log['waktu_masuk'] = log['waktu_masuk'].strftime('%H:%M:%S')  # Jika objek datetime

            if isinstance(log['waktu_keluar'], timedelta):
                log['waktu_keluar'] = str(log['waktu_keluar'])  # Ubah timedelta ke string
            elif log['waktu_keluar']:
                log['waktu_keluar'] = log['waktu_keluar'].strftime('%H:%M:%S')  # Jika objek datetime
        
        df = pd.DataFrame(logs)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Absensi')
        output.seek(0)
        
        # Mengirimkan file Excel sebagai respons
        return Response(output.getvalue(),
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        headers={"Content-Disposition": "attachment; filename=log_absen.xlsx"})

    # Mengirimkan data absensi ke template 'log_absen_admin.html'
    return render_template('log_absen_admin.html', logs=logs)


@app.route('/log-absen', methods=['GET'])
def log_absen():
    # Menghubungkan ke database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Mengambil data absensi dari database, hanya ambil jam waktu masuk dan keluar
    cursor.execute("""
        SELECT a.id, a.id_pegawai, a.nama, a.tanggal, 
               TIME(a.waktu_masuk) AS waktu_masuk, 
               TIME(a.waktu_keluar) AS waktu_keluar, 
               p.nama AS nama 
        FROM absensi a 
        JOIN pegawai p ON a.id_pegawai = p.id_pegawai
    """)
    logs = cursor.fetchall()

    cursor.close()
    conn.close()

    # Mengirimkan data absensi ke template 'log_absen.html'
    return render_template('log_absen.html', logs=logs, page_css='css/log_absen.css')


# Route for absen page
# @app.route('/absen')
# def absen():
#     return render_template('absenn.html')

# Route for add dataset
@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        data = request.json
        image_data = data.get('image_data')

        if not image_data:
            return jsonify({'success': False, 'message': 'Gambar tidak ditemukan'}), 400

        try:
            img_data = base64.b64decode(image_data)
            print("Decoding berhasil")
        except Exception as e:
            print(f"Decoding gagal: {e}")
            return jsonify({'success': False, 'message': 'Decoding gagal'}), 400

        # Load LBPHFaceRecognizer and Haar Cascade
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("recognizer/trainingdata.yml")  # Path ke model pelatihan

        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'message': 'Format gambar tidak didukung'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dalam gambar
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        print(f"Jumlah wajah terdeteksi: {len(faces)}")

        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'Wajah tidak terdeteksi'}), 404

        matched = False
        employee_name = None
        employee_id = None
        highest_accuracy = 0
        alert_message = None

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = gray[y:y + h, x:x + w]

            # Prediksi wajah
            id, conf = recognizer.predict(face)
            print(f"Prediksi: ID={id}, Confidence={conf}")

            if conf < 100:  # Sesuaikan threshold untuk confidence
                matched = True
                highest_accuracy = 100 - conf
                alert_message = "Wajah cocok dengan dataset."
                employee_id = id
                break
            else:
                alert_message = "Wajah tidak cocok dengan dataset."

        if matched:
            today_date = datetime.date.today()
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            # Ambil nama pegawai dari tabel pegawai
            cursor.execute("SELECT nama FROM pegawai WHERE id_pegawai = %s", (employee_id,))
            pegawai = cursor.fetchone()

            if not pegawai:
                cursor.close()
                conn.close()
                return jsonify({
                    'success': False,
                    'message': 'Pegawai tidak ditemukan dalam database.',
                    'accuracy': round(highest_accuracy, 2),
                    'alert': alert_message,
                }), 404

            employee_name = pegawai['nama']

            # Cek apakah pegawai sudah absen hari ini
            cursor.execute("""
                SELECT * FROM absensi 
                WHERE id_pegawai = %s AND tanggal = %s
            """, (employee_id, today_date))
            attendance_record = cursor.fetchone()

            if attendance_record:
                # Absensi keluar
                cursor.execute("""
                    UPDATE absensi
                    SET waktu_keluar = NOW()
                    WHERE id_pegawai = %s AND tanggal = %s
                """, (employee_id, today_date))
                message = f'Absensi keluar berhasil untuk {employee_name}'
            else:
                # Absensi masuk
                cursor.execute("""
                    INSERT INTO absensi (id_pegawai, nama, tanggal, waktu_masuk)
                    VALUES (%s, %s, %s, NOW())
                """, (employee_id, employee_name, today_date))
                message = f'Absensi masuk berhasil untuk {employee_name}'

            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({
                'success': True,
                'message': message,
                'employee': employee_name,
                'accuracy': round(highest_accuracy, 2),
                'alert': alert_message,
            }), 201
        else:
            return jsonify({
                'success': False,
                'message': 'Wajah tidak cocok dengan dataset',
                'accuracy': round(highest_accuracy, 2),
                'alert': alert_message,
            }), 404

    return render_template('absen.html', page_css='css/absen.css')

@app.route('/add-dataset', methods=['GET', 'POST'])
def add_dataset():
    if request.method == 'POST':
        data = request.json
        nama_pegawai = data.get('nama_pegawai')
        nama_image_prefix = data.get('nama_image_prefix')
        image_batch = data.get('image_batch')

        if not nama_pegawai or not image_batch:
            return jsonify({'success': False, 'message': 'Nama pegawai dan gambar diperlukan'}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            # Cari pegawai di database
            cursor.execute("SELECT id_pegawai, nama FROM pegawai WHERE nama = %s", (nama_pegawai,))
            pegawai = cursor.fetchone()

            if not pegawai:
                return jsonify({'success': False, 'message': 'Pegawai tidak ditemukan'}), 404

            id_pegawai = pegawai['id_pegawai']
            formatted_name = nama_pegawai.lower().replace(" ", "_")
            dataset_dir = os.path.join('static', 'datasets', formatted_name)
            os.makedirs(dataset_dir, exist_ok=True)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            saved_images = []

            for i, image_data in enumerate(image_batch):
                try:
                    img_data = base64.b64decode(image_data)
                    np_arr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as e:
                    return jsonify({'success': False, 'message': f'Gagal memproses gambar ke-{i + 1}: {e}'}), 400

                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                for (x, y, w, h) in faces_detected:
                    cropped_face = gray[y:y + h, x:x + w]
                    image_path = os.path.join(dataset_dir, f'{nama_image_prefix}_{i + 1}.jpg')
                    cv2.imwrite(image_path, cropped_face)

                    saved_images.append({
                        'filename': f'{nama_image_prefix}_{i + 1}.jpg',
                        'path': image_path
                    })

                    cursor.execute("""
                        INSERT INTO dataset (id_pegawai, nama_image, image_path)
                        VALUES (%s, %s, %s)
                    """, (id_pegawai, f'{nama_image_prefix}_{i + 1}.jpg', image_path))
                    break  # Ambil hanya wajah pertama yang terdeteksi

            conn.commit()

            # Proses pelatihan model setelah dataset ditambahkan
            train_model()

            return jsonify({'success': True, 'message': 'Dataset berhasil disimpan dan model telah dilatih', 'images': saved_images}), 201

        except mysql.connector.Error as err:
            conn.rollback()
            return jsonify({'success': False, 'message': f"Database error: {err}"}), 500
        finally:
            cursor.close()
            conn.close()

    return render_template('add_dataset.html')


def train_model():
    # Koneksi ke database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Query data pegawai
    cursor.execute("SELECT id_pegawai, nama FROM pegawai")
    pegawai_database = cursor.fetchall()

    cursor.close()
    conn.close()

    dataset_path = "static/datasets/"
    label_map_path = "recognizer/label_map.json"
    training_data_path = "recognizer/trainingdata.yml"

    label_map = {}
    faces = []
    labels = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for pegawai in pegawai_database:
        id_pegawai = pegawai['id_pegawai']
        nama_pegawai = pegawai['nama']

        formatted_name = nama_pegawai.lower().replace(" ", "_")
        folder_path = os.path.join(dataset_path, formatted_name)

        if os.path.exists(folder_path):
            label_map[formatted_name] = id_pegawai

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                for (x, y, w, h) in faces_detected:
                    faces.append(img[y:y + h, x:x + w])
                    labels.append(id_pegawai)

    # Simpan label_map ke file JSON
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=4)

    if len(faces) == 0:
        print("Dataset tidak memiliki wajah untuk dilatih.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
    recognizer.write(training_data_path)

    print("Model berhasil dilatih dan disimpan.")


def train_model():
    # Koneksi ke database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Query data pegawai
    cursor.execute("SELECT id_pegawai, nama FROM pegawai")
    pegawai_database = cursor.fetchall()

    cursor.close()
    conn.close()

    dataset_path = "static/datasets/"
    label_map_path = "recognizer/label_map.json"
    training_data_path = "recognizer/trainingdata.yml"

    label_map = {}
    faces = []
    labels = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for pegawai in pegawai_database:
        id_pegawai = pegawai['id_pegawai']
        nama_pegawai = pegawai['nama']

        formatted_name = nama_pegawai.lower().replace(" ", "_")
        for folder_name in os.listdir(dataset_path):
            if folder_name.startswith(f"{id_pegawai}_") or folder_name == formatted_name:
                label_map[folder_name] = id_pegawai

                folder_path = os.path.join(dataset_path, folder_name)
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    for (x, y, w, h) in faces_detected:
                        faces.append(img[y:y + h, x:x + w])
                        labels.append(id_pegawai)

    # Simpan label_map ke file JSON
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=4)

    if len(faces) == 0:
        print("Dataset tidak memiliki wajah untuk dilatih.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
    recognizer.write(training_data_path)

    print("Model berhasil dilatih dan disimpan.")

@app.route('/validate-pegawai', methods=['POST'])
def validate_pegawai():
    data = request.json
    nama_pegawai = data.get('nama_pegawai')

    if not nama_pegawai:
        return jsonify({'success': False, 'message': 'Nama pegawai harus diisi'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id_pegawai FROM pegawai WHERE nama = %s", (nama_pegawai,))
        pegawai = cursor.fetchone()

        if not pegawai:
            return jsonify({'success': False, 'message': 'Pegawai tidak ditemukan'}), 404

        return jsonify({'success': True, 'message': 'Pegawai valid'}), 200
    except mysql.connector.Error as err:
        return jsonify({'success': False, 'message': f"Database error: {err}"}), 500
    finally:
        cursor.close()
        conn.close()



# Endpoint untuk menambahkan data admin
@app.route('/tambah-admin', methods=['POST'])
def tambah_admin():
    try:
        # Ambil data dari request JSON
        data = request.json
        username = data['username']
        email = data['email']
        password = data['password']

        # Validasi input
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'Semua field harus diisi!'}), 400

        # Koneksi ke database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Periksa apakah email sudah ada
        cursor.execute("SELECT id_admin FROM admin WHERE email = %s", (email,))
        existing_email = cursor.fetchone()

        if existing_email:
            return jsonify({'success': False, 'message': 'Email sudah digunakan!'}), 400

        # Hash password menggunakan Werkzeug
        hashed_password = generate_password_hash(password)

        # Simpan ke database
        cursor.execute("""
            INSERT INTO admin (username, email, password)
            VALUES (%s, %s, %s)
        """, (username, email, hashed_password))
        
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'success': True, 'message': 'Data admin berhasil ditambahkan'}), 201
    
    except mysql.connector.Error as err:
        return jsonify({'success': False, 'message': f'Error: {err}'}), 500

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# Route for admin: tambah pegawai
@app.route('/tambah-pegawai', methods=['GET', 'POST'])
def tambah_pegawai():
    if request.method == 'POST':
        # Cek jika admin sudah login
        if 'user_id' not in session:
            return redirect(url_for('login'))

        data = request.json
        nama = data['nama']
        nip = data['nip']
        nomor_telepon = data['nomor_telepon']
        alamat = data['alamat']

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO pegawai (nama, nip, nomor_telepon, alamat)
                VALUES (%s, %s, %s, %s)
            """, (nama, nip, nomor_telepon, alamat))
            conn.commit()
            cursor.close()
            conn.close()
            return jsonify({'success': True, 'message': 'Pegawai berhasil ditambahkan'}), 201
        except mysql.connector.Error as err:
            conn.rollback()
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': f"Error: {err}"}), 500

    # GET request: Render halaman tambah pegawai
    return render_template('add_pegawai.html')



# Route untuk grafik absensi
@app.route('/grafik-absensi', methods=['GET'])
def grafik_absensi():
    try:
        # Menghubungkan ke database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Query untuk mengambil data grafik
        cursor.execute("""
            SELECT DATE_FORMAT(tanggal, '%Y-%m-%d') as tanggal, COUNT(*) as total_absen
            FROM absensi 
            GROUP BY tanggal
            ORDER BY tanggal ASC
        """)
        grafik_data = cursor.fetchall()

    except Exception as e:
        # Logging jika terjadi error
        print(f"Error fetching grafik data: {e}")
        grafik_data = []
    finally:
        # Menutup koneksi database
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

    # Kirim data ke template
    return render_template('grafik_absensi.html', grafik_data=grafik_data)


@app.route('/pegawai', methods=['GET'])
def daftar_pegawai():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Ambil daftar pegawai dari database
    cursor.execute("SELECT * FROM pegawai")
    pegawai_list = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('pegawai.html', pegawai_list=pegawai_list, page_css='css/pegawai.css')


@app.route('/pegawai/edit/<int:id>', methods=['POST'])
def edit_pegawai(id):
    data = request.json
    nama = data.get('nama')
    nip = data.get('nip')
    nomor_telepon = data.get('nomor_telepon')
    alamat = data.get('alamat')

    if not nama or not nip or not nomor_telepon or not alamat:
        return jsonify({'success': False, 'message': 'Semua kolom wajib diisi'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE pegawai 
            SET nama = %s, nip = %s, nomor_telepon = %s, alamat = %s 
            WHERE id_pegawai = %s
        """, (nama, nip, nomor_telepon, alamat, id))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Pegawai berhasil diperbarui'}), 200
    except mysql.connector.Error as e:
        conn.rollback()
        return jsonify({'success': False, 'message': f'Database error: {e}'}), 500


@app.route('/pegawai/delete/<int:id>', methods=['POST'])
def delete_pegawai(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM pegawai WHERE id_pegawai = %s", (id,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Pegawai berhasil dihapus'}), 200
    except mysql.connector.Error as e:
        conn.rollback()
        return jsonify({'success': False, 'message': f'Database error: {e}'}), 500



if __name__ == '__main__':
    app.run(debug=True)

