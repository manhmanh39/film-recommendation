import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 1. CÁC LỚP EMBEDDING & PHỤ TRỢ ──────────────────────────────────────────
class BERT4RecEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, dropout=0.1):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        return self.dropout(self.tok_embedding(idx) + self.pos_embedding(positions))

class MetaBERT4RecEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, num_genres, dropout=0.1):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, genres):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        genre_emb = torch.matmul(genres.float(), self.genre_embedding.weight) 
        return self.dropout(self.tok_embedding(idx) + self.pos_embedding(positions) + genre_emb)

class FrequencyFilter(nn.Module):
    def __init__(self, cutoff=3):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, x):
        B, T, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        mask = torch.zeros_like(x_fft)
        mask[:, :self.cutoff, :] = 1
        return torch.fft.irfft(x_fft * mask, n=T, dim=1, norm="ortho")

class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model * 4)
        self.l2 = nn.Linear(d_model * 4, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.l2(self.activation(self.l1(x)))

# ─── 2. CÁC KHỐI TRANSFORMER (STANDARD & NOVA) ───────────────────────────────
class Trm(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mh = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = FFN(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mh(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))
        return self.ln2(x + self.dropout(self.ffn(x)))

class NovaTrm(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mh = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_pure, h_side, key_padding_mask=None):
        attn_out, _ = self.mh(h_side, h_side, h_pure, key_padding_mask=key_padding_mask)
        h_pure = self.ln1(h_pure + self.dropout(attn_out))
        h_pure = self.ln2(h_pure + self.dropout(self.ffn(h_pure)))
        g = torch.sigmoid(self.gate(h_pure))
        h_side = g * h_side + (1 - g) * h_pure
        return h_pure, h_side

# ─── 3. MÔ HÌNH HOÀN CHỈNH ──────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, max_len, d_model, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = BERT4RecEmbedding(d_model, max_len, vocab_size, dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, idx, key_padding_mask=None, candidates=None):
        T = idx.shape[1]
        x = self.embedding(idx)
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=key_padding_mask)
        x = self.ln(x)
        
        # Xử lý trả về theo Phase (Train vs Eval)
        if not self.training:
            x_last = x[:, -1, :] 
            if candidates is not None:
                return torch.gather(self.output(x_last), dim=1, index=candidates)
            return self.output(x_last) # Full Ranking
            
        return self.output(x) # Training (Full Sequence)

class BERT4Rec(nn.Module):
    def __init__(self, max_len, d_model, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = BERT4RecEmbedding(d_model, max_len, vocab_size, dropout)
        self.trm_layers = nn.ModuleList([Trm(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, d_model)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, idx, key_padding_mask=None, candidates=None):
        h = self.embedding(idx)
        for layer in self.trm_layers:
            h = layer(h, key_padding_mask=key_padding_mask)
            
        if not self.training:
            z = F.gelu(self.proj(h[:, -1, :]))
            if candidates is not None:
                c_emb = self.embedding.tok_embedding(candidates)
                logits = torch.matmul(z.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1)
                return logits + self.bias[candidates]
            return torch.matmul(z, self.embedding.tok_embedding.weight.T) + self.bias
            
        z = F.gelu(self.proj(h))
        return torch.matmul(z, self.embedding.tok_embedding.weight.T) + self.bias

class MetaBERT4Rec(nn.Module):
    def __init__(self, max_len, num_genres, d_model, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.id_emb = BERT4RecEmbedding(d_model, max_len, vocab_size, dropout)
        self.side_emb = MetaBERT4RecEmbedding(d_model, max_len, vocab_size, num_genres, dropout)
        self.freq_filter = FrequencyFilter(cutoff=3)
        self.layers = nn.ModuleList([NovaTrm(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, d_model)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, idx, genres, key_padding_mask=None, candidates=None):
        h_p = self.id_emb(idx)
        h_s = self.side_emb(idx, genres)
        h_p, h_s = self.freq_filter(h_p), self.freq_filter(h_s)
        
        for layer in self.layers:
            h_p, h_s = layer(h_p, h_s, key_padding_mask)
            
        if not self.training:
            z = F.gelu(self.proj(h_p[:, -1, :]))
            if candidates is not None:
                c_emb = self.id_emb.tok_embedding(candidates)
                logits = torch.matmul(z.unsqueeze(1), c_emb.transpose(1, 2)).squeeze(1)
                return logits + self.bias[candidates]
            return torch.matmul(z, self.id_emb.tok_embedding.weight.T) + self.bias
            
        z = F.gelu(self.proj(h_p))
        return torch.matmul(z, self.id_emb.tok_embedding.weight.T) + self.bias