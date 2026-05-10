import sqlite3
import pandas as pd
import hashlib
from datetime import datetime

# ==========================================
# DATABASE SQLITE SETUP & QUERIES
# ==========================================

def init_db():
    """Initialize database and tables if they do not exist"""
    conn = sqlite3.connect('recsys_demo.db')
    c = conn.cursor()
    # User Table (ADDED preferred_genres COLUMN)
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, 
                  password TEXT,
                  preferred_genres TEXT)''')
                  
    # Watch History Table
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  movie_id INTEGER, 
                  movie_title TEXT, 
                  watch_time DATETIME)''')
    conn.commit()
    
    # [AUTO-UPDATE OLD DB] If your database already has a users table but lacks the preferred_genres column
    try:
        c.execute("ALTER TABLE users ADD COLUMN preferred_genres TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass # Skip if column already exists

    return conn

def hash_password(password):
    """Hash password for basic security"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password):
    """Add new user to the database"""
    conn = init_db()
    c = conn.cursor()
    try:
        # When creating a new account, leave preferred_genres empty
        c.execute("INSERT INTO users (username, password, preferred_genres) VALUES (?, ?, ?)", 
                  (username, hash_password(password), ""))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Username already exists
    finally:
        conn.close()

def login_user(username, password):
    """Verify login credentials"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
              (username, hash_password(password)))
    data = c.fetchone()
    conn.close()
    return data

def log_watch_history(username, movie_id, movie_title):
    """Save user's watch history"""
    conn = init_db()
    c = conn.cursor()
    c.execute("INSERT INTO history (username, movie_id, movie_title, watch_time) VALUES (?, ?, ?, ?)", 
              (username, movie_id, movie_title, datetime.now()))
    conn.commit()
    conn.close()

def get_user_history(username):
    """Get watch history list to feed into Model/UI"""
    conn = init_db()
    
    # [FIX KEYERROR]: Use "AS movieId" to force the column name to match app.py 100%
    # [FIX SEQUENTIAL]: Use "ORDER BY watch_time ASC" to order the sequence from oldest to newest
    query = "SELECT movie_id AS movieId, movie_title, watch_time FROM history WHERE username=? ORDER BY watch_time ASC"
    
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def update_user_genres(username, genres_list):
    """Save user's favorite genres list to DB (Separated by |)"""
    conn = init_db()
    c = conn.cursor()
    genres_str = "|".join(genres_list)
    c.execute("UPDATE users SET preferred_genres = ? WHERE username = ?", (genres_str, username))
    conn.commit()
    conn.close()

def get_user_genres(username):
    """Get user's favorite genres list from DB"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT preferred_genres FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    
    if data and data[0]:
        return data[0].split("|")
    return []