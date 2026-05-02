import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, idx):  # B,T
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)  # T
        positions = positions.unsqueeze(0).expand(B, T)  # B,T
        return self.pos_embedding(positions)


class GenreEmbedding(nn.Module):
    def __init__(self, num_genres, d_model):
        super().__init__()

        self.embedding = nn.Embedding(
            num_genres,
            d_model,
        )

    def forward(self, genres):  # B, T, G (multi-hot: 0/1)
        # genres: binary indicators

        # B,T,G -> B,T,G,d
        emb = self.embedding.weight  # G,d
        emb = emb.unsqueeze(0).unsqueeze(0)  # 1,1,G,d

        genres = genres.unsqueeze(-1)  # B,T,G,1

        genres_emb = emb * genres  # mask active genres
        genres_emb = genres_emb.sum(dim=2)  # B,T,d

        return genres_emb


class BERT4RecEmbedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, dropout=0.1):
        super().__init__()

        self.tok_embedding = nn.Embedding(
            vocab_size,  # include pad &
            d_model,
            padding_idx=0,  # CRITICAL
        )

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

        self.tok_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0,  # CRITICAL
        )

        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.genre_embedding = GenreEmbedding(num_genres, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, genres):
        B, T = idx.shape

        positions = torch.arange(T, device=idx.device)
        positions = positions.unsqueeze(0).expand(B, T)

        tok_emb = self.tok_embedding(idx)  # B,T,d
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
        attn_out, _ = self.mh(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        x = self.layer_norm(x)

        pffn_out = self.pffn(x)
        x = x + self.dropout(pffn_out)
        x = self.layer_norm(x)

        return x


class BERT4Rec(nn.Module):
    def __init__(
        self, max_len, d_model, n_heads, n_layers, vocab_size, dropout=0.1
    ):
        super().__init__()

        self.embedding = BERT4RecEmbedding(
            d_model, max_len, vocab_size, dropout=dropout
        )
        self.trm_layers = nn.ModuleList(
            [Trm(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.proj = nn.Linear(d_model, d_model)  # a = Wa + b
        self.gelu = nn.GELU()
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(
        self,
        idx,
        key_padding_mask,
        candidates=None,
    ):
        B, _ = idx.shape
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


class MetaBERT4Rec(nn.Module):
    def __init__(
        self,
        max_len,
        num_genres,
        d_model,
        n_heads,
        n_layers,
        vocab_size,
        dropout=0.1,
    ):
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
        self.proj = nn.Linear(d_model, d_model)  # a = Wa + b
        self.gelu = nn.GELU()
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(
        self,
        idx,
        genres,
        key_padding_mask,
        candidates=None,
    ):
        B, _ = idx.shape

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


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm

#     ds = MovieLenDataset(
#         movies=movies,
#         ratings=ratings,
#         max_len=max_len,
#         min_len=min_len,
#         split="train",
#     )

#     loader = DataLoader(
#         dataset=ds,
#         batch_size=4,
#         shuffle=True,
#         num_workers=2,
#     )

#     b = next(iter(loader))

#     model = MetaBERT4Rec(
#         max_len=200,
#         d_model=64,
#         n_heads=4,
#         n_layers=6,
#         num_genres=18,
#         vocab_size=27279,
#     )

#     model.to("cuda")

#     out = model.forward(
#         idx=b["input"],
#         genres=b["genres"],
#         token_mask=b["token_mask"],
#         key_padding_mask=b["key_padding_mask"],
#         candidates=b["candidates"],
#     )

#     out.shape
