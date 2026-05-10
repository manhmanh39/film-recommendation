import sqlite3
import pandas as pd
import hashlib
from datetime import datetime

# ==========================================
# DATABASE SQLITE SETUP & QUERIES
# ==========================================

def init_db():
    """Khởi tạo database và các bảng nếu chưa có"""
    conn = sqlite3.connect('recsys_demo.db')
    c = conn.cursor()
    # Bảng User
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT)''')
    # Bảng Lịch sử xem phim
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  movie_id INTEGER, 
                  movie_title TEXT, 
                  watch_time DATETIME)''')
    conn.commit()
    return conn

def hash_password(password):
    """Mã hóa mật khẩu để bảo mật cơ bản"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password):
    """Thêm người dùng mới vào database"""
    conn = init_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Trùng username
    finally:
        conn.close()

def login_user(username, password):
    """Kiểm tra thông tin đăng nhập"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
              (username, hash_password(password)))
    data = c.fetchone()
    conn.close()
    return data

def log_watch_history(username, movie_id, movie_title):
    """Lưu lịch sử xem phim của user"""
    conn = init_db()
    c = conn.cursor()
    c.execute("INSERT INTO history (username, movie_id, movie_title, watch_time) VALUES (?, ?, ?, ?)", 
              (username, movie_id, movie_title, datetime.now()))
    conn.commit()
    conn.close()

def get_user_history(username):
    """Lấy danh sách lịch sử xem phim để đưa vào Model/UI"""
    conn = init_db()
    
    # SỬA LẠI: Không dùng f-string để ghép chuỗi trực tiếp
    query = "SELECT movie_id, movie_title, watch_time FROM history WHERE username=? ORDER BY watch_time DESC"
    
    # Sử dụng tham số (params) an toàn của Pandas
    df = pd.read_sql_query(query, conn, params=(username,))
    
    conn.close()
    return df