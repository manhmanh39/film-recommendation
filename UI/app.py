import sys
import os
import streamlit as st
import pandas as pd
import database as db
import requests
import torch
import numpy as np

# ==========================================
# AUTOMATIC PATH CONFIGURATION (ABSOLUTE PATHS)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # Currently in UI directory
parent_dir = os.path.dirname(current_dir) # Go back to root directory
src_dir = os.path.join(parent_dir, 'src') # Enter src directory

if src_dir not in sys.path:
    sys.path.append(src_dir)

from model import SASRecF_Concat 

# Point accurately to the model file
MODEL_PATH = os.path.join(parent_dir, "data", "sasrec_f_64_meta_timesplit", "best_model.pt") 

# Point accurately to the data files
MOVIES_PATH = os.path.join(parent_dir, "data", "ml-32m", "movies.csv")
LINKS_PATH = os.path.join(parent_dir, "data", "ml-32m", "links.csv")

# Hyperparameters
D_MODEL = 64
MAX_LEN = 200
N_HEADS = 2
N_LAYERS = 2
VOCAB_SIZE = 87587
NUM_GENRES = 20

# List of 20 standard MovieLens genres
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance", 
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"
]

@st.cache_resource(show_spinner="Initializing Artificial Intelligence...")
def load_ml_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SASRecF_Concat(
            max_len=MAX_LEN, num_genres=NUM_GENRES, 
            d_model=D_MODEL, n_heads=N_HEADS, 
            n_layers=N_LAYERS, vocab_size=VOCAB_SIZE
        ).to(device)
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        return None, "cpu"

# ==========================================
# FUNCTION TO FETCH POSTERS FROM TMDB
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_poster(tmdb_id):
    default_poster = "https://via.placeholder.com/500x750?text=No+Poster"
    if pd.isna(tmdb_id):
        return default_poster
        
    api_key = "7eba21d33de55ed49b040bbdc2a18e08" 
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=en-US"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except:
        pass
    return default_poster

# ==========================================
# CSS CONFIGURATION (NETFLIX VIBE)
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #141414; color: #FFFFFF; }
        div.stButton > button:first-child {
            background-color: #E50914; color: #FFFFFF; border: none;
            border-radius: 4px; padding: 10px 24px; font-weight: bold; width: 100%; transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover { background-color: #B20710; }
        .stTabs [data-baseweb="tab-list"] { justify-content: center; }
        .login-header {
            text-align: center; font-size: 3.5rem; font-weight: 900; color: #E50914;
            margin-bottom: 1rem; letter-spacing: 3px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# DATASET PROCESSING AND ID MAPPING
# ==========================================
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv(MOVIES_PATH)
        links = pd.read_csv(LINKS_PATH)
        df = pd.merge(movies, links, on='movieId', how='left')
        
        df['movie_idx'] = df.index + 1 
        movie2idx = dict(zip(df['movieId'], df['movie_idx']))
        idx2movie = dict(zip(df['movie_idx'], df['movieId']))
        
        return df, movie2idx, idx2movie
    except FileNotFoundError:
        st.error(f"❌ Data not found! Please check the directory: {MOVIES_PATH}")
        return pd.DataFrame(), {}, {}

# ==========================================
# AI INFERENCE FUNCTION
# ==========================================
def get_ai_recommendations(history_df, df, movie2idx, idx2movie, model, device, top_k=8):
    if model is None or history_df.empty:
        return df.sample(top_k) 
    
    history_movie_ids = history_df['movieId'].tolist()
    seq_idx = [movie2idx[mid] for mid in history_movie_ids if mid in movie2idx]
    
    if not seq_idx:
        return df.sample(top_k)
        
    seq_idx = seq_idx[-MAX_LEN:]
    pad_len = MAX_LEN - len(seq_idx)
    input_tensor = torch.tensor([0] * pad_len + seq_idx, dtype=torch.long).unsqueeze(0).to(device)
    
    dummy_genres = torch.zeros((1, MAX_LEN, NUM_GENRES), dtype=torch.float).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor, dummy_genres)
        if len(predictions.shape) == 3: 
            logits = predictions[0, -1, :] 
        else: 
            logits = predictions[0]
            
    for idx in seq_idx:
        if idx < len(logits):
            logits[idx] = -float('inf')
    logits[0] = -float('inf') 
            
    scores, top_indices = torch.topk(logits, top_k)
    top_indices = top_indices.cpu().numpy().tolist()
    
    recommended_movieIds = [idx2movie[idx] for idx in top_indices if idx in idx2movie]
    recommended_df = df[df['movieId'].isin(recommended_movieIds)].copy()
    
    recommended_df['rank'] = pd.Categorical(recommended_df['movieId'], categories=recommended_movieIds, ordered=True)
    recommended_df = recommended_df.sort_values('rank').drop('rank', axis=1)
    
    return recommended_df

# ==========================================
# UI HELPER: RENDER MOVIE GRID
# ==========================================
def render_movie_grid(movies_df, section_key, num_columns=4):
    """Helper function to render the movie grid for reuse, avoiding code duplication"""
    cols = st.columns(num_columns)
    for index, row in movies_df.reset_index().iterrows():
        with cols[index % num_columns]:
            with st.container(border=True):
                st.image(fetch_poster(row['tmdbId']), use_container_width=True)
                st.markdown(f"<div style='height: 60px; overflow: hidden; margin-top: 5px;'><b>{row['title']}</b></div>", unsafe_allow_html=True)
                if st.button("▶ Watch Now", key=f"btn_{section_key}_{row['movieId']}", use_container_width=True):
                    db.log_watch_history(st.session_state['username'], row['movieId'], row['title'])
                    st.session_state['open_link'] = f"https://www.themoviedb.org/movie/{int(row['tmdbId'])}"
                    if 'recommended_movies' in st.session_state:
                        del st.session_state['recommended_movies'] 
                    st.rerun()

# ==========================================
# MAIN INTERACTIVE UI
# ==========================================
def main():
    st.set_page_config(page_title="Flow Cine AI", page_icon="🎬", layout="wide")
    inject_custom_css()
    db.init_db()

    ai_model, device = load_ml_model()

    if 'open_link' in st.session_state:
        js_code = f"window.open('{st.session_state['open_link']}', '_blank');"
        st.components.v1.html(f"<script>{js_code}</script>", height=0)
        del st.session_state['open_link']

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    df, movie2idx, idx2movie = load_data()

    # --- LOGIN / REGISTER SCREEN ---
    if not st.session_state['logged_in']:
        st.markdown('<div class="login-wrapper login-header">FLOW CINE</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
            with tab1:
                log_user = st.text_input("Username")
                log_pass = st.text_input("Password", type='password')
                if st.button("LOGIN"):
                    if db.login_user(log_user, log_pass):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = log_user
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")
            with tab2:
                reg_user = st.text_input("Username", key="reg")
                reg_pass = st.text_input("Password", type='password', key="regp")
                if st.button("REGISTER"):
                    if db.add_user(reg_user, reg_pass): 
                        # [NEW FEATURE] Auto-login after registration
                        st.success("✅ Registration successful! Logging in automatically...")
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = reg_user
                        st.session_state['onboarding_complete'] = False
                        st.rerun()

    # --- MAIN SCREEN ---
    else:
        def go_home():
            st.session_state['search_input'] = ""
            if 'recommended_movies' in st.session_state:
                del st.session_state['recommended_movies']

        # Sidebar
        with st.sidebar:
            st.markdown(f"## 👤 Hello,\n### **{st.session_state['username']}**")
            history_df = db.get_user_history(st.session_state['username'])
            st.dataframe(history_df[['movie_title']] if not history_df.empty else pd.DataFrame(), hide_index=True, width='stretch')
            if st.button("Logout", width='stretch'):
                st.session_state['logged_in'] = False
                st.rerun()

        # Header
        col_title, col_logo = st.columns([5, 1.5]) 
        with col_title:
            search_query = st.text_input("🔍 Enter movie name...", key="search_input")
        with col_logo:
            st.button("🏠 Home", on_click=go_home, use_container_width=True)

        st.divider()

        if not df.empty:
            # --- 1. SEARCH SCREEN ---
            if st.session_state.get('search_input', ''):
                st.markdown(f"### 🔎 Results for: {st.session_state['search_input']}")
                results = df[df['title'].str.contains(st.session_state['search_input'], case=False, na=False)].head(15)
                for index, row in results.iterrows():
                    col1, col2, col3 = st.columns([1,3,1])
                    with col1: st.image(fetch_poster(row['tmdbId']))
                    with col2: st.write(row['title'])
                    with col3:
                        if st.button("Watch Now", key=f"src_{row['movieId']}", use_container_width=True):
                            db.log_watch_history(st.session_state['username'], row['movieId'], row['title'])
                            st.session_state['open_link'] = f"https://www.themoviedb.org/movie/{int(row['tmdbId'])}"
                            st.rerun()
            
            # --- 2. HOME SCREEN ---
            else:
                # Fetch genre preferences from Database
                saved_genres = db.get_user_genres(st.session_state['username'])
                
                # [ONBOARDING FLOW FOR BRAND NEW USERS] (No watch history + No saved genres)
                if history_df.empty and not saved_genres:
                    st.markdown("<h2 style='text-align: center; color: #E50914;'>Welcome to Flow Cine! 🍿</h2>", unsafe_allow_html=True)
                    st.markdown("<h4 style='text-align: center;'>To help our AI make the best recommendations, please select your favorite movie genres:</h4><br>", unsafe_allow_html=True)
                    
                    _, col_center, _ = st.columns([1, 2, 1])
                    with col_center:
                        selected_genres = st.multiselect("Select genres (Multiple choices allowed):", ALL_GENRES)
                        st.write("")
                        if st.button("Start exploring with AI", use_container_width=True):
                            if not selected_genres:
                                st.warning("⚠️ Please select at least 1 genre to continue!")
                            else:
                                # SAVE TO DATABASE PERMANENTLY
                                db.update_user_genres(st.session_state['username'], selected_genres)
                                st.rerun() # Reload page
                                
                # [MOVIE ROWS DISPLAY FLOW]
                else:
                    # ROW 1: RECOMMENDATIONS
                    if history_df.empty:
                        # User selected genres but hasn't watched any movies (Cold-start)
                        st.markdown("### 🎯 Movies in your favorite genres")
                        st.caption("Based on your profile, we recommend the following blockbusters:")
                        
                        # [BUG FIX]: Save movies to session_state to prevent randomizing on button click
                        if 'cold_start_movies' not in st.session_state:
                            pattern = '|'.join(saved_genres)
                            genre_movies = df[df['genres'].str.contains(pattern, case=False, na=False)]
                            st.session_state['cold_start_movies'] = genre_movies.sample(min(8, len(genre_movies)))
                            
                        render_movie_grid(st.session_state['cold_start_movies'], "cold_start", num_columns=4)
                        
                    else:
                        # User clicked a movie -> Activate Personalized SASRec Model
                        st.markdown("### ✨ For You")
                        st.caption("🤖 Our AI analyzed your watch history and predicts you will like:")
                        
                        if 'recommended_movies' not in st.session_state:
                            with st.spinner('AI is calculating your preference matrix...'):
                                st.session_state['recommended_movies'] = get_ai_recommendations(
                                    history_df=history_df, df=df, movie2idx=movie2idx, idx2movie=idx2movie, 
                                    model=ai_model, device=device, top_k=8
                                )
                        render_movie_grid(st.session_state['recommended_movies'], "ai_recs", num_columns=4)

                    st.divider()
                    
                    # ROW 2: MOST WATCHED
                    st.markdown("### 🏆 Most watched movies of all time")
                    # [BUG FIX]: Save state
                    if 'most_watched_movies' not in st.session_state:
                        st.session_state['most_watched_movies'] = df.sample(4)
                    render_movie_grid(st.session_state['most_watched_movies'], "most_watched", num_columns=4)

                    st.divider()

                    # ROW 3: TRENDING
                    st.markdown("### 🔥 Trending this week")
                    # [BUG FIX]: Save state
                    if 'trending_movies' not in st.session_state:
                        st.session_state['trending_movies'] = df.sample(4)
                    render_movie_grid(st.session_state['trending_movies'], "trending", num_columns=4)

if __name__ == '__main__':
    main()