from flask import Flask, render_template, request, redirect, url_for, session, flash
from concurrent.futures import ThreadPoolExecutor
import pymysql.cursors
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import pandas as pd
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import io
import zipfile
import markdown
import nltk
from tfidf import tfidf
from gemini import gemini
from bert import bert
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from pprint import pprint
import re

# Inisialisasi model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='cek_cv',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
app = Flask(__name__)

app.config["SECRET_KEY"] = "iniSecretKeyKu2019"
nltk.data.path.append('C:/nltk_data')
nltk.download('punkt')

# Tentukan direktori untuk menyimpan file yang diunggah sementara
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    """Ekstraksi teks dari PDF"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return None

def get_embedding(text):
    """Mendapatkan embedding menggunakan BERT"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def calculate_similarity(cv_text, jd_embedding):
    """Menghitung kesamaan antara CV dan JD"""
    cv_embedding = get_embedding(cv_text)
    similarity_score = cosine_similarity(cv_embedding, jd_embedding).flatten()
    return similarity_score[0]

# def highlight_text(cv_text, keywords):
#     """Highlight keywords in the CV text."""
#     highlighted_text = cv_text
#     for keyword in keywords:
#         # Gunakan regex untuk menemukan keyword secara case-insensitive dan tambahkan highlight
#         highlighted_text = re.sub(
#             rf'({re.escape(keyword)})', 
#             r'<span style="background-color: yellow">\1</span>', 
#             highlighted_text, 
#             flags=re.IGNORECASE
#         )
#     return highlighted_text

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if request.method == 'POST':
        # Upload CV files
        cv_files = request.files.getlist('cv[]')
        job_description_file = request.files['job_description']

        saved_cv_paths = []
        for cv_file in cv_files:
            if cv_file and allowed_file(cv_file.filename):
                cv_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cv_file.filename))
                cv_file.save(cv_path)
                saved_cv_paths.append(cv_path)

        if job_description_file and allowed_file(job_description_file.filename):
            jd_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(job_description_file.filename))
            job_description_file.save(jd_path)

            # Ekstrak teks dari job description dan dapatkan embeddingnya
            jd_text = extract_text_from_pdf(jd_path)
            jd_embedding = get_embedding(jd_text)

            # Ekstraksi keywords dari Job Description (misalnya dengan split atau TF-IDF)
            # keywords = jd_text.split()  # Ini contoh sederhana; bisa gunakan TF-IDF atau NLP untuk hasil lebih baik

            # Analisis CV menggunakan ThreadPoolExecutor
            # Analisis CV
            similarities = {}
            cv_texts = {}
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_cv, cv_path, jd_embedding): cv_path for cv_path in saved_cv_paths}
                for future in futures:
                    cv_path = futures[future]
                    try:
                        similarity, cv_text = future.result()
                        similarities[cv_path] = similarity
                        # Highlight keywords dalam CV
                        cv_texts[cv_path] = cv_text
                    except Exception as exc:
                        print(f"{cv_path} generated an exception: {exc}")

            # Cari CV dengan similarity tertinggi
            best_match = max(similarities, key=similarities.get)
            best_similarity_score = similarities[best_match]
            best_similarity_score_percent = best_similarity_score * 100
            best_cv_text = cv_texts[best_match]

            # Simpan hasil analisis ke database
            try:
                with connection.cursor() as cursor:
                    sql = "INSERT INTO history (job_description, best_cv_path, similarity_score, user_id, best_cv_text) VALUES (%s, %s, %s, %s, %s)"
                    cursor.execute(sql, (jd_text, best_match, best_similarity_score_percent, session['user_id'], best_cv_text))
                    connection.commit()
                flash("Scan completed successfully!")
            except Exception as e:
                flash(f"Failed to save scan result: {e}", "danger")

            return render_template(
                'result.html',
                best_match=best_match,
                best_similarity_score_percent=best_similarity_score_percent,
                best_cv_text=best_cv_text
            )
        
    return render_template('scan.html')

def process_cv(cv_path, jd_embedding):
    """Memproses setiap CV dan menghitung kesamaan dengan JD serta menyimpan isi CV"""
    cv_text = extract_text_from_pdf(cv_path)
    if cv_text:
        similarity_score = calculate_similarity(cv_text, jd_embedding)
        return similarity_score, cv_text  # Return similarity score dan isi CV
    return 0, ""  # Return 0 dan teks kosong jika terjadi kesalahan

@app.route("/history")
def history():
    user_id = session.get('user_id')  # Mengambil user_id dari sesi, jika tidak ada, nilainya None
    results = []

    if user_id:
        try:
            with connection.cursor() as cursor:
                # Query untuk mengambil jumlah scan per tanggal, dikelompokkan berdasarkan user dan tanggal
                sql = """
                    SELECT DATE(created_at) AS scan_date, COUNT(*) AS total_scan
                    FROM history
                    WHERE user_id = %s
                    GROUP BY DATE(created_at)
                    ORDER BY scan_date DESC
                """
                cursor.execute(sql, (user_id,))
                results = cursor.fetchall()  # Mengambil semua hasil query sebagai list of dicts
        except Exception as e:
            flash(f"Error retrieving scan history: {e}", "danger")
    
    return render_template("history.html", results=results)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/send-contact', methods=['POST'])
def send_contact():
    try:
        nama = request.form['nama']
        email = request.form['email']
        teks = request.form['teks']

        # Simpan ke database
        with connection.cursor() as cursor:
            sql = "INSERT INTO messages (nama, email, teks) VALUES (%s, %s, %s)"
            cursor.execute(sql, (nama, email, teks))
            connection.commit()

        flash('Pesan berhasil dikirim!', 'success')
        return redirect(url_for('contact'))
    except Exception as e:
        flash(f'Gagal mengirimkan pesan: {e}', 'danger')
        return redirect(url_for('contact'))

@app.route("/login")
def login():
    return render_template("login.html")

@app.route('/req-login', methods=['POST'])
def req_login():
    email = request.form['email']
    password = request.form['password']

    try:
        with connection.cursor() as cursor:
            # Cari user berdasarkan email
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user['password'], password):
                # Jika password cocok
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['nama'] = user['nama']
                session['email'] = user['email']
                session['role'] = user['role']
                flash('Login berhasil!', 'success')
                
                if user['role'] == 'Admin':
                    return redirect(url_for('dashboard'))
                else:
                    return redirect(url_for('index'))
            else:
                flash('Email atau password salah', 'danger')
                return redirect(url_for('login'))
    except Exception as e:
        flash(f'Error saat login: {e}', 'danger')
        return redirect(url_for('login'))

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/req-register", methods=["POST"])
def req_register():
    try:
        nama = request.form['nama']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        nama_perusahaan = request.form['nama_perusahaan']
        jumlah_karyawan = request.form['jumlah_karyawan']
        role = request.form['role']

        # Hash password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        # Simpan ke database
        with connection.cursor() as cursor:
            sql = "INSERT INTO users (nama, username, email, password, nama_perusahaan, jumlah_karyawan, role) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (nama, username, email, hashed_password, nama_perusahaan, jumlah_karyawan, role))
            connection.commit()

        flash('Registrasi berhasil! Silakan Login.', 'success')
        return redirect(url_for('login'))
    except Exception as e:
        flash(f'Registrasi Gagal: {e}', 'danger')
        return redirect(url_for('users'))

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'email' not in session:
                flash('Anda harus login untuk mengakses halaman ini.', 'danger')
                return redirect(url_for('login'))
            elif role and session.get('role') != role:
                flash('Anda tidak memiliki akses ke halaman ini.', 'danger')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/logout")
def logout():
    session.clear()  # Menghapus seluruh session
    flash("Anda telah berhasil logout.", "success")
    return redirect(url_for('login'))

@app.route("/admin/dashboard")
@login_required(role="Admin")
def dashboard():
    return render_template("admin/dashboard.html")

# ADMIN - Users
@app.route("/admin/add-user")
@login_required(role="Admin")
def add_user_page():
    return render_template("admin/add-user.html")

@app.route('/admin/users')
@login_required(role="Admin")
def users():
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM users"
            cursor.execute(sql)
            result = cursor.fetchall()  # result diharapkan menjadi list of dictionaries
            return render_template("admin/users.html", users=result)  # Kirim data 'users' ke template
    except Exception as e:
        return render_template("admin/users.html", users=[], error=str(e))  # Kirim error ke template jika gagal

@app.route('/admin/create-user', methods=['POST'])
@login_required(role="Admin")
def create_user():
    try:
        nama = request.form['nama']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        # Hash password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        # Simpan ke database
        with connection.cursor() as cursor:
            sql = "INSERT INTO users (nama, username, email, password, role) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql, (nama, username, email, hashed_password, role))
            connection.commit()

        flash('User berhasil ditambahkan!', 'success')
        return redirect(url_for('users'))
    except Exception as e:
        flash(f'Gagal menambahkan user: {e}', 'danger')
        return redirect(url_for('users'))

@app.route('/admin/update-user', methods=['POST'])
@login_required(role="Admin")
def update_user():
    try:
        user_id = request.form['user_id']
        nama = request.form['nama']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Pengondisian Password - Jika password diisi, hash password baru
        if password:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
            update_sql = "UPDATE users SET nama = %s, username = %s, email = %s, password = %s WHERE id = %s"
            data = (nama, username, email, hashed_password, user_id)
        else:
            update_sql = "UPDATE users SET nama = %s, username = %s, email = %s WHERE id = %s"
            data = (nama, username, email, user_id)

        # Eksekusi SQL untuk update user
        with connection.cursor() as cursor:
            cursor.execute(update_sql, data)
            connection.commit()

        flash('User berhasil diperbarui!', 'success')
        return redirect(url_for('users'))
    except Exception as e:
        flash(f'Gagal memperbarui user: {e}', 'danger')
        return redirect(url_for('users'))

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@login_required(role="Admin")
def delete_user(user_id):
    try:
        # Eksekusi query untuk menghapus user berdasarkan ID
        with connection.cursor() as cursor:
            sql = "DELETE FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            connection.commit()

        flash('User berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus user: {e}', 'danger')
    return redirect(url_for('users'))

# ADMIN - Messages
@app.route('/admin/messages')
@login_required(role="Admin")
def pesan():
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM messages"
            cursor.execute(sql)
            result = cursor.fetchall() 
            return render_template("admin/message.html", pesan=result) 
    except Exception as e:
        return render_template("admin/message.html", pesan=[], error=str(e))

@app.route('/admin/delete-msg/<int:msg_id>', methods=['POST'])
@login_required(role="Admin")
def delete_msg(msg_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM messages WHERE id = %s"
            cursor.execute(sql, (msg_id,))
            connection.commit()

        flash('Message berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus message: {e}', 'danger')
    return redirect(url_for('pesan'))

# ADMIN - Scan History
@app.route('/admin/scan-history')
@login_required(role="Admin")
def scan_history():
    try:
        with connection.cursor() as cursor:
            sql = """
            SELECT history.*, users.username, users.nama, users.email 
            FROM history
            JOIN users ON history.user_id = users.id
            """
            cursor.execute(sql)
            result = cursor.fetchall() 
            pprint(result)
            return render_template("admin/scan_history.html", scan_history=result) 
    except Exception as e:
        pprint(e)
        return render_template("admin/scan_history.html", scan_history=[], error=str(e))
    
@app.route('/admin/delete-scan/<int:scan_id>', methods=['POST'])
@login_required(role="Admin")
def delete_scan(scan_id):
    try:
        with connection.cursor() as cursor:
            sql = "DELETE FROM history WHERE id = %s"
            cursor.execute(sql, (scan_id,))
            connection.commit()

        flash('Scan History berhasil dihapus!', 'success')
    except Exception as e:
        flash(f'Gagal menghapus scan history: {e}', 'danger')
    return redirect(url_for('scan_history'))

@app.route('/admin/detail-scan/<int:scan_id>', methods=['GET'])
@login_required(role="Admin")
def detail_scan(scan_id):
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM history WHERE id = %s"
            cursor.execute(sql, (scan_id,))
            result = cursor.fetchone()

            # Jika data tidak ditemukan
            if not result:
                flash(f"Detail scan dengan ID {scan_id} tidak ditemukan!", "error")
                return redirect(url_for('scan_history'))

            # Kirim data ke template
            return render_template("admin/detail_scan.html", history=result)

    except Exception as e:
        # Tangani error dengan pesan dan arahkan kembali
        flash(f"Terjadi kesalahan: {str(e)}", "error")
        return redirect(url_for('scan_history'))

if __name__ == "__main__":
    app.run(debug=True, port=5001)