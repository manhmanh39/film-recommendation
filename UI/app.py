import streamlit as st
import pandas as pd
import database as db
import requests

# ==========================================
# [THÊM MỚI] HÀM LẤY ẢNH TỪ TMDB
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_poster(tmdb_id):
    # Đường link ảnh mặc định nếu phim không có ảnh hoặc lỗi
    default_poster = "https://via.placeholder.com/500x750?text=No+Poster"
    
    if pd.isna(tmdb_id):
        return default_poster
        
    # THAY API KEY CỦA BẠN VÀO ĐÂY
    api_key = "7eba21d33de55ed49b040bbdc2a18e08" 
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}&language=vi-VN"
    
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            # Lấy link ảnh chuẩn của TMDb (w500 là kích thước ảnh)
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except Exception as e:
        pass # Nếu có lỗi mạng hoặc API, sẽ bỏ qua và dùng ảnh mặc định
        
    return default_poster
# ==========================================
# CẤU HÌNH CSS (NETFLIX VIBE)
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* Đổi màu nền của toàn bộ ứng dụng sang màu tối */
        .stApp {
            background-color: #141414;
            color: #FFFFFF;
        }
        /* Style cho nút bấm chính sang màu đỏ Netflix */
        div.stButton > button:first-child {
            background-color: #E50914;
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            padding: 10px 24px;
            font-weight: bold;
            width: 100%;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #B20710;
            color: #FFFFFF;
            border-color: #B20710;
        }
        /* Đưa thẻ Tabs ra giữa */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        /* Style form đăng nhập */
        .login-header {
            text-align: center;
            font-size: 3.5rem;
            font-weight: 900;
            color: #E50914;
            margin-bottom: 1rem;
            letter-spacing: 3px;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# XỬ LÝ DATASET
# ==========================================
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv('movies.csv')
        links = pd.read_csv('links.csv')
        df = pd.merge(movies, links, on='movieId', how='left')
        return df
    except FileNotFoundError:
        st.error("Không tìm thấy file movies.csv hoặc links.csv!")
        return pd.DataFrame()

# ==========================================
# GIAO DIỆN UI TƯƠNG TÁC
# ==========================================
def main():
    # Cấu hình layout full màn hình
    st.set_page_config(page_title="Hệ thống Gợi ý Phim", page_icon="🎬", layout="wide")
    inject_custom_css()
    db.init_db()

    # ==========================================
    # [THÊM MỚI] XỬ LÝ MỞ LINK PHIM TỰ ĐỘNG
    # ==========================================
    if 'open_link' in st.session_state:
        # Dùng JavaScript để ép trình duyệt mở link trong tab mới
        js_code = f"window.open('{st.session_state['open_link']}', '_blank');"
        st.components.v1.html(f"<script>{js_code}</script>", height=0)
        
        # Xóa link khỏi bộ nhớ để tránh bị mở lại khi tương tác với các nút khác
        del st.session_state['open_link']

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    df = load_data()

# ==========================================
    # MÀN HÌNH 1: LANDING PAGE ĐĂNG NHẬP (Ở GIỮA)
    # ==========================================
    if not st.session_state['logged_in']:
        import base64
        
        @st.cache_data(show_spinner=False)
        def get_base64_file(file_path):
            try:
                with open(file_path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception as e:
                return ""

        # Hãy chắc chắn tên này giống hệt 100% với tên file trong thư mục của bạn
        bg_image_file = "poster_background.jpg" 
        bg_base64 = get_base64_file(bg_image_file)
        logo_base64 = get_base64_file("logo.png")

        # THÊM BÁO LỖI: Báo cho bạn biết nếu gọi sai tên file ảnh
        if not bg_base64:
            st.error(f"❌ Không tìm thấy file ảnh: '{bg_image_file}'. Hãy kiểm tra lại tên file hoặc đường dẫn!")

        # 1. CHÈN BACKGROUND ẢNH VÀ STYLE CHO KHUNG FORM CÓ BLUR
        st.markdown(f"""
            <style>
            .stApp {{
                background-color: transparent !important;
            }}
            
            header {{
                visibility: hidden;
            }}
            .block-container {{
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
                max-width: 100% !important;
            }}
            
            /* [ĐÃ SỬA] Cấu hình lại kích thước và độ mờ của ảnh nền */
            #background-image {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;   /* Đưa về chuẩn 100% chiều rộng màn hình */
                height: 100vh;  /* Đưa về chuẩn 100% chiều cao màn hình */
                z-index: -100;
                object-fit: cover;
                
                /* Mẹo: Phóng to nhẹ 2% để giấu viền mờ (blur) mà không tạo thanh cuộn */
                transform: scale(1.02); 
                
                /*  blur 2px, tăng sáng một chút cho rõ ảnh */
                filter: brightness(0.5) blur(2px); 
                background-color: #141414; 
            }}
            
            [data-testid="column"]:nth-of-type(2) {{
                background-color: rgba(0, 0, 0, 0.85); 
                padding: 1.5rem 2.5rem; 
                border-radius: 12px;
                box-shadow: 0px 8px 24px rgba(0, 0, 0, 0.9);
            }}
            .login-wrapper {{
                margin-top: 2vh; 
                text-align: center;
                margin-bottom: 1rem;
            }}
            </style>
            
            <img id="background-image" src="data:image/jpeg;base64,{bg_base64}">
        """, unsafe_allow_html=True)
        # 2. HIỂN THỊ LOGO TRƯỚC FORM
        if logo_base64:
            st.markdown(f"""
                <div class="login-wrapper">
                    <img src="data:image/png;base64,{logo_base64}" style="max-width: 220px;">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="login-wrapper login-header">FLOW CINE</div>', unsafe_allow_html=True)
        
        # 3. GIAO DIỆN FORM ĐĂNG NHẬP
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col2:
            st.markdown("<h3 style='text-align: center; color: #fff; margin-bottom: 20px;'>Đăng nhập để trải nghiệm</h3>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký tài khoản"])
            
            with tab1:
                log_username = st.text_input("Tên đăng nhập", key="log_user")
                log_password = st.text_input("Mật khẩu", type='password', key="log_pass")
                if st.button("ĐĂNG NHẬP"):
                    if db.login_user(log_username, log_password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = log_username
                        st.rerun() # Lệnh này sẽ load lại trang và thoát khỏi khối 'if not logged_in', tắt video ngay lập tức!
                    else:
                        st.error("❌ Sai tên đăng nhập hoặc mật khẩu.")
            
            with tab2:
                reg_username = st.text_input("Tên đăng nhập", key="reg_user")
                reg_password = st.text_input("Mật khẩu", type='password', key="reg_pass")
                reg_password_confirm = st.text_input("Xác nhận mật khẩu", type='password', key="reg_pass_conf")
                if st.button("ĐĂNG KÝ"):
                    if reg_password != reg_password_confirm:
                        st.error("❌ Mật khẩu xác nhận không khớp.")
                    elif db.add_user(reg_username, reg_password):
                        st.success("✅ Tạo tài khoản thành công! Hãy quay lại tab Đăng nhập.")
                    else:
                        st.error("❌ Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.")

    # ==========================================
    # MÀN HÌNH 2: TRANG CHỦ PHIM (KHI ĐÃ ĐĂNG NHẬP)
    # ==========================================
    else:
        # --- BƯỚC 1: THÊM HÀM CALLBACK CHO NÚT TRANG CHỦ ĐỂ FIX LỖI ---
        def go_home():
            # Xử lý reset session_state trước khi UI kịp render lại
            st.session_state['search_input'] = ""
            if 'recommended_movies' in st.session_state:
                del st.session_state['recommended_movies']
            if 'active_movie' in st.session_state:
                del st.session_state['active_movie']

        # Thanh sidebar quản lý Profile
        with st.sidebar:
            st.markdown(f"## 👤 Xin chào,\n### **{st.session_state['username']}**")
            st.divider()
            
            st.markdown("### 🕒 Lịch sử xem phim")
            history_df = db.get_user_history(st.session_state['username'])
            if history_df.empty:
                st.info("Chưa có dữ liệu.")
            else:
                # Fix cảnh báo Terminal: Đổi use_container_width=True thành width='stretch'
                st.dataframe(history_df[['movie_title']], hide_index=True, width='stretch')
            
            st.divider()
            if st.button("Đăng xuất", key="logout_btn", width='stretch'):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ''
                st.rerun()

        # Khu vực chính - Duyệt phim
        import base64

        # Hàm đọc file ảnh sang base64 để nhúng vào CSS
        def get_base64_image(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
            except Exception as e:
                # Hiển thị lỗi ra màn hình để dễ debug nếu không tìm thấy file
                st.error(f"Không thể tải logo: {e}")
                return ""

        # Lấy dữ liệu ảnh (Sử dụng đường dẫn tương đối, tên file đã đổi ngắn gọn)
        logo_base64 = get_base64_image("logo.png")
        
       # CSS tùy chỉnh để biến nút bấm thành Logo
        if logo_base64: # Chỉ render CSS khi đã đọc được ảnh
            st.markdown(f"""
                <style>
                /* 1. ẨN TAG CHỮ "go_home_logo" KHI HOVER */
                div[data-testid="stTooltipContent"] {{
                    display: none !important;
                }}
                
                /* 2. CHỈNH NÚT THÀNH LOGO VÀ HIỂN THỊ TRỌN VẸN ẢNH */
                div[data-testid="stTooltipHoverTarget"] button {{
                    background-image: url('data:image/png;base64,{logo_base64}') !important;
                    
                    /* BẮT BUỘC DÙNG contain ĐỂ HIỂN THỊ ĐẦY ĐỦ KHÔNG BỊ CẮT XÉN */
                    background-size: contain !important; 
                    
                    background-repeat: no-repeat !important;
                    background-position: center !important; /* Đưa logo ra giữa khung nút */
                    background-color: transparent !important;
                    border: none !important;
                    
                    /* Set cứng kích thước khung nút để chứa ảnh to hơn */
                    width: 120px !important; 
                    height: 120px !important; 
                    
                    box-shadow: none !important;
                    padding: 0 !important;
                    
                    /* Bỏ scale lớn cũ đi vì dễ gây lỗi vỡ layout */
                    transform: none !important; 
                    
                    /* Đẩy logo xuống một chút cho cân đối với ô tìm kiếm bên cạnh */
                    margin-top: 5px !important; 
                }}
                
                /* Ẩn chữ "Home" mặc định bên trong nút */
                div[data-testid="stTooltipHoverTarget"] button p {{
                    display: none !important;
                }}
                
                /* Hiệu ứng nảy nhẹ lên khi di chuột vào Logo */
                div[data-testid="stTooltipHoverTarget"] button:hover {{
                    transform: scale(1.1) !important; 
                    background-color: transparent !important;
                    border: none !important;
                }}
                </style>
            """, unsafe_allow_html=True)

        # Chia layout: Cột lớn cho Tiêu đề + Search, Cột nhỏ ở góc phải cho Logo
        col_title, col_logo = st.columns([5, 1.5]) 
        
        with col_title:
            st.markdown("<h1>🎬 Khám phá nội dung</h1>", unsafe_allow_html=True)
            search_query = st.text_input("🔍 Nhập tên phim bạn muốn tìm kiếm...", key="search_input")
            
        with col_logo:
            # Nút bấm ẩn được CSS bắt qua lớp bọc tooltip để thay bằng ảnh
            st.button("Home", help="go_home_logo", on_click=go_home, key="logo_btn_home")

        st.divider()

        if not df.empty:
            # ==========================================
            # TRẠNG THÁI 1: MÀN HÌNH KẾT QUẢ TÌM KIẾM (LIST VIEW)
            # ==========================================
            if st.session_state.get('search_input', ''):
                st.markdown(f"### 🔎 Kết quả tìm kiếm cho: <span style='color:#E50914'>{st.session_state['search_input']}</span>", unsafe_allow_html=True)
                results = df[df['title'].str.contains(st.session_state['search_input'], case=False, na=False)].head(15)
                
                if results.empty:
                    st.warning("Không tìm thấy bộ phim nào phù hợp với từ khóa của bạn.")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Dùng dạng List (Danh sách dọc) cho kết quả tìm kiếm
                    # Dùng dạng List (Danh sách dọc) cho kết quả tìm kiếm
                    for index, row in results.iterrows():
                        with st.container():
                            # CHIA THÀNH 3 CỘT: Ảnh (1) - Thông tin (3) - Nút (1)
                            col_img, col_info, col_btn = st.columns([1, 3, 1]) 
                            
                            with col_img:
                                # Gọi hàm lấy ảnh
                                poster_url = fetch_poster(row['tmdbId'])
                                st.image(poster_url, use_container_width=True)
                                
                            with col_info:
                                st.markdown(f"**{row['title']}**")
                                st.caption(f"🏷️ Thể loại: `{row['genres']}`")
                                
                            with col_btn:
                                # Căn giữa nút bấm theo chiều dọc
                                st.write("") 
                                button_key = f"watch_search_{row['movieId']}"
                                if st.button("▶ Xem ngay", key=button_key, use_container_width=True):
                                    db.log_watch_history(st.session_state['username'], row['movieId'], row['title'])
                                    if pd.notna(row['tmdbId']):
                                        st.session_state['open_link'] = f"https://www.themoviedb.org/movie/{int(row['tmdbId'])}"
                                    else:
                                        st.toast("❌ Không có link nguồn cho phim này.", icon="⚠️")
                                    st.rerun()
                        st.divider()
            # ==========================================
            # TRẠNG THÁI 2: MÀN HÌNH TRANG CHỦ / GỢI Ý (GRID VIEW)
            # ==========================================
            else:
                # ---------------------------------------------------------
                # [TÍNH NĂNG MỚI] ONBOARDING KIỂU TINDER CHO NGƯỜI DÙNG MỚI
                # ---------------------------------------------------------
                # Khởi tạo các biến trạng thái cho luồng Tinder
                if 'onboarding_complete' not in st.session_state:
                    st.session_state['onboarding_complete'] = False
                if 'onboarding_index' not in st.session_state:
                    st.session_state['onboarding_index'] = 0
                if 'liked_genres' not in st.session_state:
                    st.session_state['liked_genres'] = set()
                if 'onboarding_movies' not in st.session_state:
                    # Lấy ngẫu nhiên 5 phim có poster để người dùng "quẹt"
                    st.session_state['onboarding_movies'] = df.sample(5).to_dict('records')

                # NẾU CHƯA CÓ LỊCH SỬ XEM VÀ CHƯA HOÀN THÀNH QUẸT TINDER
                if history_df.empty and not st.session_state['onboarding_complete']:
                    st.markdown("<h2 style='text-align: center; color: #E50914;'>❤️ Khám phá gu phim của bạn</h2>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; margin-bottom: 30px;'>Hãy cho chúng tôi biết bạn thích bộ phim nào dưới đây để nhận gợi ý chuẩn nhất nhé!</p>", unsafe_allow_html=True)

                    current_idx = st.session_state['onboarding_index']
                    
                    if current_idx < len(st.session_state['onboarding_movies']):
                        current_movie = st.session_state['onboarding_movies'][current_idx]
                        
                        # Chia cột để tạo giao diện Card (thẻ) ở giữa màn hình giống Tinder
                        _, col_card, _ = st.columns([1.5, 2, 1.5])
                        
                        with col_card:
                            with st.container(border=True):
                                # Ảnh phim
                                poster_url = fetch_poster(current_movie['tmdbId'])
                                st.image(poster_url, use_container_width=True)
                                
                                # Thông tin phim
                                st.markdown(f"<h3 style='text-align: center; margin-top: 10px;'>{current_movie['title']}</h3>", unsafe_allow_html=True)
                                st.caption(f"<div style='text-align: center;'>🎭 {current_movie['genres']}</div>", unsafe_allow_html=True)
                                st.write("") # Đệm
                                
                                # Hai nút Quẹt Trái / Quẹt Phải
                                btn_col1, btn_col2 = st.columns(2)
                                with btn_col1:
                                    if st.button("❌ Bỏ qua", use_container_width=True, key=f"skip_{current_idx}"):
                                        st.session_state['onboarding_index'] += 1
                                        st.rerun()
                                with btn_col2:
                                    if st.button("❤️ Thích", use_container_width=True, key=f"like_{current_idx}"):
                                        # Bóc tách thể loại của phim này và lưu vào bộ nhớ
                                        genres_list = current_movie['genres'].split('|')
                                        for g in genres_list:
                                            # Bỏ qua các thể loại rác nếu có
                                            if g != "(no genres listed)":
                                                st.session_state['liked_genres'].add(g)
                                        
                                        st.session_state['onboarding_index'] += 1
                                        st.rerun()
                                        
                                # Hiển thị tiến trình (VD: 1/5)
                                st.progress((current_idx) / len(st.session_state['onboarding_movies']))
                                st.caption(f"<div style='text-align: center;'>Phim {current_idx + 1} / {len(st.session_state['onboarding_movies'])}</div>", unsafe_allow_html=True)
                    else:
                        # Khi đã duyệt hết 5 phim, đánh dấu hoàn thành và tải lại trang
                        st.session_state['onboarding_complete'] = True
                        st.rerun()

                # ---------------------------------------------------------
                # HIỂN THỊ LƯỚI PHIM GỢI Ý (Khi đã có lịch sử HOẶC đã quẹt xong)
                # ---------------------------------------------------------
                else:
                    st.markdown("### ✨ Gợi Ý Dành Riêng Cho Bạn")
                    st.caption("🍿 Dựa trên gu điện ảnh của bạn, đây là những siêu phẩm bạn không nên bỏ lỡ!")
                    
                    if 'recommended_movies' not in st.session_state:
                        # NẾU USER LÀ NGƯỜI MỚI VỪA QUẸT TINDER XONG
                        if history_df.empty and st.session_state['onboarding_complete']:
                            if len(st.session_state['liked_genres']) > 0:
                                # Tạo chuỗi Regex từ các thể loại đã thích (VD: 'Action|Romance|Comedy')
                                genre_pattern = '|'.join(list(st.session_state['liked_genres']))
                                # Lọc các phim có chứa ít nhất 1 thể loại người dùng đã "Thích"
                                filtered_df = df[df['genres'].str.contains(genre_pattern, case=False, na=False)]
                                
                                # Nếu kho phim lọc được đủ nhiều, lấy 16 phim, nếu ít quá thì lấy tất cả
                                sample_size = min(16, len(filtered_df))
                                st.session_state['recommended_movies'] = filtered_df.sample(sample_size)
                            else:
                                # Nếu người dùng lỡ bấm "Bỏ qua" cả 5 phim, fallback về random
                                st.session_state['recommended_movies'] = df.sample(16)
                                
                        # NẾU USER CŨ ĐÃ CÓ LỊCH SỬ XEM TỪ TRƯỚC (Logic mặc định)
                        else:
                            st.session_state['recommended_movies'] = df.sample(16) 
                            
                    results = st.session_state['recommended_movies']

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Dùng dạng Grid (Lưới 4 cột) cho trang chủ
                    cols = st.columns(4)
                    for index, row in results.reset_index().iterrows():
                        with cols[index % 4]:
                            # Tạo Card bọc mỗi phim
                            with st.container(border=True):
                                poster_url = fetch_poster(row['tmdbId'])
                                st.image(poster_url, use_container_width=True)
                                
                                st.markdown(f"<div style='height: 60px; overflow: hidden; margin-bottom: 5px; margin-top: 10px;'><b>{row['title']}</b></div>", unsafe_allow_html=True)
                                st.caption(f"🎭 {row['genres']}")
                                
                                st.write("")
                                button_key = f"watch_home_{row['movieId']}"
                                
                                if st.button("▶ Xem Phim", key=button_key, use_container_width=True):
                                    db.log_watch_history(st.session_state['username'], row['movieId'], row['title'])
                                    if pd.notna(row['tmdbId']):
                                        st.session_state['open_link'] = f"https://www.themoviedb.org/movie/{int(row['tmdbId'])}"
                                    else:
                                        st.toast("❌ Không có link nguồn cho phim này.", icon="⚠️")
                                    st.rerun()
if __name__ == '__main__':
    main()