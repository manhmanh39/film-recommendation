import os
import torch
import pandas as pd
import re
from model import MetaBERT4Rec

device = "cuda" if torch.cuda.is_available() else "cpu"

class RecommendationPipeline:
    def __init__(self, model_path, data_dir="/tmp/ml-32m", max_len=200, d_model=128):
        print("⚙️ Khởi tạo Hệ thống Gợi ý MetaBERT4Rec...")
        self.device = device
        self.max_len = max_len
        
        # 1. Load Metadata phim
        movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
        
        # Trích xuất Year
        def extract_year(title):
            match = re.search(r'\((\d{4})\)', str(title))
            return int(match.group(1)) if match else 2000
        movies['year'] = movies['title'].apply(extract_year)
        
        # Lập chỉ mục phim để tra cứu nhanh
        self.movie_meta = movies.set_index('movieId').to_dict('index')
        self.vocab_size = len(movies) + 2
        
        # Mapping Genres (Bê từ class MovieLenDataset sang để inference)
        all_genres = set('|'.join(movies['genres']).split('|'))
        all_genres.discard('(no genres listed)')
        self.genre_list = sorted(list(all_genres))
        self.genre_map = {g: i for i, g in enumerate(self.genre_list)}
        self.num_genres = len(self.genre_list)
        
        # 2. Load Model
        self.model = MetaBERT4Rec(max_len, self.num_genres, d_model, 4, 4, self.vocab_size).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model"])
            self.model.eval()
            print("✅ Đã nạp thành công bộ não MetaBERT4Rec!")
        else:
            raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")

    def _get_item_features(self, movie_id):
        """Lấy genre multi-hot và year của 1 bộ phim."""
        genres_vector = [0.0] * self.num_genres
        if movie_id in self.movie_meta:
            g_str = self.movie_meta[movie_id]['genres'].split('|')
            for g in g_str:
                if g in self.genre_map:
                    genres_vector[self.genre_map[g]] = 1.0
        return genres_vector

    def recommend(self, history_ids, top_k=10):
        """Đưa ra dự đoán dựa trên lịch sử xem phim."""
        # Chuyển đổi dữ liệu sang định dạng tensor
        seq_idx = history_ids[-self.max_len:]
        seq_genres = [self._get_item_features(m_id) for m_id in seq_idx]
        
        # Thêm 1 token giả [MASK] ở cuối để yêu cầu mô hình dự đoán tương lai
        # (vocab_size - 1 thường là Mask Token, tuỳ thiết lập lúc training)
        mask_token = self.vocab_size - 1
        seq_idx.append(mask_token) 
        seq_genres.append([0.0] * self.num_genres) # Genre trống cho MASK
        
        # Padding
        pad_len = self.max_len - len(seq_idx)
        if pad_len > 0:
            seq_idx = [0] * pad_len + seq_idx
            seq_genres = [[0.0] * self.num_genres] * pad_len + seq_genres
            
        idx_tensor = torch.tensor([seq_idx], dtype=torch.long).to(self.device)
        genres_tensor = torch.tensor([seq_genres], dtype=torch.float32).to(self.device)
        
        # Key padding mask: True ở vị trí padding (ID = 0)
        padding_mask = (idx_tensor == 0)

        # Suy luận
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = self.model(idx_tensor, genres_tensor, key_padding_mask=padding_mask, candidates=None)
                
        # Lấy kết quả tại vị trí cuối cùng
        last_step_logits = logits[0, -1, :]
        
        # Lọc bỏ các phim người dùng đã xem rồi (không gợi ý lại)
        last_step_logits[history_ids] = -float('inf')
        last_step_logits[0] = -float('inf') # Bỏ padding
        last_step_logits[mask_token] = -float('inf') # Bỏ mask token
        
        # Lấy Top K
        scores, top_indices = torch.topk(last_step_logits, top_k)
        
        print("\n🍿 LỊCH SỬ XEM GẦN ĐÂY CỦA USER:")
        for m_id in history_ids[-5:]:
            meta = self.movie_meta.get(m_id, {"title": "Unknown", "genres": "Unknown"})
            print(f" - {meta['title']} | {meta['genres']}")
            
        print(f"\n🎯 TOP {top_k} PHIM GỢI Ý (ĐƯỢC METABERT4REC LỰA CHỌN):")
        for i, (m_id, score) in enumerate(zip(top_indices.tolist(), scores.tolist())):
            meta = self.movie_meta.get(m_id, {"title": "Unknown", "genres": "Unknown"})
            print(f"{i+1:2d}. [Điểm: {score:5.2f}] | {meta['title'][:40].ljust(40)} | {meta['genres']}")

# ==========================================
# TEST PIPELINE DEMO
# ==========================================
if __name__ == "__main__":
    d_model = 128
    meta_model_path = f"../data/metabert4rec_{d_model}/best_model.pt"
    
    try:
        pipeline = RecommendationPipeline(model_path=meta_model_path)
        
        # Giả lập một User yêu thích phim Khoa học viễn tưởng & Phiêu lưu (Sci-Fi, Adventure)
        # IDs: 260 (Star Wars: IV), 1196 (Star Wars: V), 1210 (Star Wars: VI), 2571 (The Matrix)
        user_history = [260, 1196, 1210, 2571] 
        
        pipeline.recommend(user_history, top_k=10)
        
    except FileNotFoundError as e:
        print(e)
        print("Bạn cần train xong MetaBERT4Rec trước khi chạy Pipeline nhé!")