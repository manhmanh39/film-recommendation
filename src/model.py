import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, idx):  # B,T
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        positions = positions.unsqueeze(0).expand(B, T)
        return self.pos_embedding(positions)


class GenreEmbedding(nn.Module):
    def __init__(self, num_genres, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_genres, d_model)

    def forward(self, genres):  # B, T, G (multi-hot: 0/1)
        emb = self.embedding.weight  # G,d
        emb = emb.unsqueeze(0).unsqueeze(0)  # 1,1,G,d
        genres = genres.unsqueeze(-1)  # B,T,G,1
        genres_emb = emb * genres  # mask active genres
        genres_emb = genres_emb.sum(dim=2)  # B,T,d
        return genres_emb


class BERT4RecEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, dropout=0.1):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        positions = positions.unsqueeze(0).expand(B, T)
        tok_emb = self.tok_embedding(idx)
        pos_emb = self.pos_embedding(positions)
        emb = tok_emb + pos_emb
        emb = self.dropout(emb)
        return emb


class MetaBERT4RecEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, num_genres, dropout=0.1):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.genre_embedding = GenreEmbedding(num_genres, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, genres):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        positions = positions.unsqueeze(0).expand(B, T)
        tok_emb = self.tok_embedding(idx)
        pos_emb = self.pos_embedding(positions)
        genre_emb = self.genre_embedding(genres)
        emb = tok_emb + pos_emb + genre_emb
        emb = self.dropout(emb)
        return emb


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gelu = nn.GELU()
        self.l1 = nn.Linear(d_model, d_model * 4)
        self.l2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.l2(self.gelu(self.l1(x)))


class PFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = FFN(d_model)

    def forward(self, x):
        return self.ffn(x)


# ── Transformer block dùng chung cho BERT4Rec (bidirectional) ──────────────
class Trm(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mh = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pffn = PFFN(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mh(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.layer_norm(x)
        pffn_out = self.pffn(x)
        x = x + self.dropout(pffn_out)
        x = self.layer_norm(x)
        return x


# ── Causal Transformer block dùng cho SASRec (unidirectional) ──────────────
class CausalTrm(nn.Module):
    """
    Giống Trm nhưng dùng causal mask (attention mask tam giác dưới).
    SASRec chỉ nhìn vào quá khứ, không nhìn tương lai.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mh = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pffn = PFFN(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x, key_padding_mask=None):
        T = x.shape[1]

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        attn_out, _ = self.mh(
            x, x, x,
            attn_mask=causal_mask,              # bool mask
            key_padding_mask=key_padding_mask,  # giữ nguyên bool
        )

        x = x + self.dropout(attn_out)
        x = self.layer_norm(x)

        pffn_out = self.pffn(x)
        x = x + self.dropout(pffn_out)
        x = self.layer_norm(x)

        return x


# ── BERT4Rec ────────────────────────────────────────────────────────────────
class BERT4Rec(nn.Module):
    def __init__(self, max_len, d_model, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = BERT4RecEmbedding(d_model, max_len, vocab_size, dropout=dropout)
        self.trm_layers = nn.ModuleList(
            [Trm(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.proj = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, idx, key_padding_mask, candidates=None):
        h = self.embedding(idx)
        for layer in self.trm_layers:
            h = layer(h, key_padding_mask=key_padding_mask)

        if candidates is not None:
            h_last = h[:, -1, :]
            z = self.gelu(self.proj(h_last))
            candidates_embedding = self.embedding.tok_embedding(candidates)
            logits = torch.matmul(
                z.unsqueeze(1), candidates_embedding.transpose(1, 2)
            ).squeeze(1)
            logits = logits + self.output_bias[candidates]
        else:
            z = self.gelu(self.proj(h))
            logits = torch.matmul(z, self.embedding.tok_embedding.weight.T)
            logits = logits + self.output_bias

        return logits


# ── MetaBERT4Rec ─────────────────────────────────────────────────────────────
class MetaBERT4Rec(nn.Module):
    def __init__(self, max_len, num_genres, d_model, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = MetaBERT4RecEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=vocab_size,
            num_genres=num_genres,
            dropout=dropout,
        )
        self.trm_layers = nn.ModuleList(
            [Trm(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.proj = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, idx, genres, key_padding_mask, candidates=None):
        h = self.embedding(idx, genres)
        for layer in self.trm_layers:
            h = layer(h, key_padding_mask=key_padding_mask)

        if candidates is not None:
            h_last = h[:, -1, :]
            z = self.gelu(self.proj(h_last))
            candidates_embedding = self.embedding.tok_embedding(candidates)
            logits = torch.matmul(
                z.unsqueeze(1), candidates_embedding.transpose(1, 2)
            ).squeeze(1)
            logits = logits + self.output_bias[candidates]
        else:
            z = self.gelu(self.proj(h))
            logits = torch.matmul(z, self.embedding.tok_embedding.weight.T)
            logits = logits + self.output_bias

        return logits


# ── SASRec ───────────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, max_len, d_model, n_heads, n_layers, vocab_size, dropout=0.1):
        super().__init__()

        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 🔥 QUAN TRỌNG: zero padding embedding
        with torch.no_grad():
            self.tok_embedding.weight[0].fill_(0)

    def forward(self, idx, key_padding_mask=None, candidates=None):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)

        x = self.tok_embedding(idx) + self.pos_embedding(pos)
        x = self.dropout(x)

        # Causal mask (True = mask)
        causal_mask = torch.triu(
            torch.ones(T, T, device=idx.device), diagonal=1
        ).bool()

        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=key_padding_mask
            )

        x = self.layer_norm(x)

        # --- PHẦN SỬA ĐỔI ĐỂ HỖ TRỢ VALIDATION ---
        if candidates is not None:
            # Trong SASRec (causal), ta chỉ quan tâm đến hidden state ở vị trí cuối cùng (T-1)
            h_last = x[:, -1, :]  # [B, d_model]
            
            # Tính full logits tại bước cuối
            full_logits = self.output(h_last) # [B, vocab_size]
            
            # Lấy ra điểm của các phim ứng với candidates
            # candidates thường có dạng [B, num_candidates]
            logits = torch.gather(full_logits, dim=1, index=candidates)
            return logits
        # -----------------------------------------

        logits = self.output(x)  # [B, T, V]
        return logits