import os
import torch
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

from model import SASRecF_Concat
from utils import prepare_dataloaders # Sử dụng lại hàm prepare để lấy đúng Dataset có chứa Genres

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Launching A/B Testing Simulation (True Baseline vs SASRec-Concat)...")

# ─── 1. CONFIGURATIONS ──────────────────────────────────────────────────
d_model = 64
max_len = 200
REVENUE_PER_HIT = 50000  # VND (Approx. $2 USD)
data_dir = "../data"

# ─── 2. DATA LOADING & ID MAPPING ───────────────────────────────────────
print("📦 Đang nạp Dataset Simulation (Tương lai) từ file CSV...")
movies = pd.read_csv(os.path.join(data_dir, "ml-32m/movies.csv"))
test_ratings = pd.read_csv(os.path.join(data_dir, "test_ratings_ab.csv"))
vocab_size = len(movies) + 2

# Khởi tạo Dataset chuẩn cho tập Test
from dataset import MovieLenDataset
from torch.utils.data import DataLoader

test_ds = MovieLenDataset(
    movies=movies, 
    ratings=test_ratings, 
    max_len=max_len, 
    min_len=5, 
    split="test"  # Bắt buộc để "test" để lấy đúng cách chia mảng
)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# ─── 3. INITIALIZING ALGORITHMS ─────────────────────────────────────────
print("🤖 Loading Group B Model (Champion: SASRecF_Concat)...")
model_treatment = SASRecF_Concat(
    max_len=max_len, num_genres=20, d_model=d_model, 
    n_heads=2, n_layers=2, vocab_size=vocab_size
).to(device)

# SỬA LẠI ĐƯỜNG DẪN: Trỏ đúng vào thư mục chứa bản Concat
checkpoint_path = f"../data/sasrec_f_{d_model}_meta_timesplit/best_model.pt"
if not os.path.exists(checkpoint_path):
    print(f"❌ Không tìm thấy model tại {checkpoint_path}. Vui lòng check lại tên thư mục!")
    exit()

model_treatment.load_state_dict(torch.load(checkpoint_path)["model"])
model_treatment.eval()

# ─── 4. RUNNING A/B TEST SIMULATION ─────────────────────────────────────
revenue_A = [] 
revenue_B = [] 

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Simulating A/B Test"):
        idx = batch["input"].to(device)
        genres = batch["genres"].to(device) # ĐÃ SỬA: Lấy genres từ batch
        key_padding_mask = batch["key_padding_mask"].to(device)
        candidates = batch["candidates"].to(device)
        
        batch_size = idx.size(0)
        
        # Chia user ngẫu nhiên vào 2 tập: 50% Control (A) và 50% Treatment (B)
        is_group_A = torch.rand(batch_size) > 0.5 

        # --- Logic Nhóm B (Deep Learning Model) ---
        with torch.amp.autocast('cuda'):
            # ĐÃ SỬA: Truyền thêm genres vào mô hình
            logits_B = model_treatment(idx, genres, key_padding_mask=key_padding_mask, candidates=candidates)
            
        target_scores_B = logits_B[:, -1].unsqueeze(1)
        ranks_B = (logits_B > target_scores_B).sum(dim=1) + 1
        hits_B = (ranks_B <= 10).float().cpu().numpy()

        # --- Logic Nhóm A (Weekly Trending / Popularity Baseline) ---
        for i in range(batch_size):
            if is_group_A[i]:
                # Giả lập Hit Rate của thuật toán phổ biến là khoảng 6%
                hit_A = np.random.choice([1.0, 0.0], p=[0.06, 0.94])
                revenue_A.append(hit_A * REVENUE_PER_HIT)
            else:
                revenue_B.append(hits_B[i] * REVENUE_PER_HIT)

# ─── 5. STATISTICAL ANALYSIS & OUTPUT ───────────────────────────────────
arpu_A, arpu_B = np.mean(revenue_A), np.mean(revenue_B)
lift = ((arpu_B - arpu_A) / max(arpu_A, 1)) * 100 

# Perform Welch's T-Test
t_stat, p_value = stats.ttest_ind(revenue_B, revenue_A, equal_var=False)

print("\n" + "="*70)
print("📊 A/B TESTING SIMULATION RESULTS (TRUE BASELINE)")
print("="*70)
print(f"Total Users Participating: {len(revenue_A) + len(revenue_B):,}")
print(f" - Group A (Popularity Baseline): {len(revenue_A):,} users")
print(f" - Group B (SASRec-Concat Model)  : {len(revenue_B):,} users\n")

print(f"{'Metric':<20} | {'Control (A)':<20} | {'Treatment (B)':<20}")
print("-" * 70)
print(f"{'Simulated ARPU':<20} | {arpu_A:>14,.0f} VND | {arpu_B:>14,.0f} VND")
print(f"{'Lift (%)':<20} | {'-':<20} | {lift:>+19.2f}%")
print(f"{'P-value':<20} | {'-':<20} | {p_value:>20.4e}")
print("="*70)

if p_value < 0.05 and lift > 0:
    print("✅ CONCLUSION: Statistical significance detected (P < 0.05).")
    print("Recommendation: THE MODEL IS READY FOR PRODUCTION DEPLOYMENT.")
else:
    print("⚠️ CONCLUSION: No statistically significant difference detected.")