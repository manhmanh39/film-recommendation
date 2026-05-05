import os
import torch
import pandas as pd
from model import SASRec, BERT4Rec, MetaBERT4Rec
from utils import prepare_dataloaders, validate_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Bắt đầu Benchmark trên thiết bị: {device}")

# 1. Cấu hình chung
d_model = 128
max_len = 200
batch_size = 16 # Dùng batch nhỏ để tránh OOM khi test Full Ranking

# Load dữ liệu (Chỉ lấy val_loaders và vocab_size)
_, pop_loader, rnd_loader, trd_loader, vocab_size = prepare_dataloaders(
    temp_dir="/tmp/ml-32m", max_len=max_len, min_len=5, 
    batch_size=batch_size, val_batch_size=batch_size
)

# Danh sách các mô hình cần Benchmark
models_to_test = [
    {
        "name": "SASRec",
        "path": f"../data/sasrec_ce_{d_model}/best_model.pt",
        "is_meta": False,
        "instance": SASRec(max_len, d_model, 4, 4, vocab_size).to(device)
    },
    {
        "name": "BERT4Rec",
        "path": f"../data/bert4rec_{d_model}/best_model.pt",
        "is_meta": False,
        "instance": BERT4Rec(max_len, d_model, 4, 4, vocab_size).to(device)
    },
    {
        "name": "MetaBERT4Rec",
        "path": f"../data/metabert4rec_{d_model}/best_model.pt",
        "is_meta": True,
        "instance": MetaBERT4Rec(max_len, 18, d_model, 4, 4, vocab_size).to(device)
    }
]

results = []

for m in models_to_test:
    name = m["name"]
    model = m["instance"]
    ckpt_path = m["path"]
    is_meta = m["is_meta"]
    
    print(f"\n" + "="*40)
    print(f"🔍 Đang đánh giá: {name}")
    print("="*40)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy trọng số tại {ckpt_path}. Bỏ qua.")
        continue
        
    # Load trọng số
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model.eval()
    
    # Đánh giá trên 3 tập
    ndcg_pop = validate_epoch(model, pop_loader, "Popularity", device, is_meta)
    ndcg_rnd = validate_epoch(model, rnd_loader, "Random", device, is_meta)
    ndcg_trd = validate_epoch(model, trd_loader, "Trending", device, is_meta)
    
    results.append({
        "Model": name,
        "NDCG@10 (Random)": round(ndcg_rnd, 4),
        "NDCG@10 (Popularity)": round(ndcg_pop, 4),
        "NDCG@10 (Trending)": round(ndcg_trd, 4),
        "Avg NDCG@10": round((ndcg_pop + ndcg_rnd + ndcg_trd) / 3, 4)
    })

# Xuất kết quả
df_results = pd.DataFrame(results)
print("\n" + "🏆 BẢNG TỔNG SẮP KẾT QUẢ BENCHMARK (FULL RANKING) 🏆".center(60))
print(df_results.to_markdown(index=False))

# Lưu ra file CSV để làm báo cáo
df_results.to_csv("../data/benchmark_results.csv", index=False)
print("\n📁 Đã lưu kết quả tại: ../data/benchmark_results.csv")