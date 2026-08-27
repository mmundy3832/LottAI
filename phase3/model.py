"""
model.py - Phase 3 Stage A models (plans/PHASE3_PLAN.md section 3, task
instructions step 1).

M1: decoder-only transformer. M2: GRU. Both share the same input/output
contract:

  forward(context, target_known) -> logits (B, NUM_DIGIT_POSITIONS, DIGIT_OUT_SIZE)

  context:      (B, W, 9) long -- tokenizer.make_windows()["context"]
  target_known: (B, 6) long    -- tokenizer.make_windows()["target_known"]
                                   (equipment columns already MASK_ID)

target_digits (the labels) is NEVER passed into either model -- only used
by the caller (train.py) to compute the loss after forward().

Embedding scheme: one nn.Embedding per token TYPE (slot, dow, machine,
ballset, digit), sized from vocab.json (tokenizer.load_vocab()). The 9
per-draw token embeddings (or 6 for target_known) are summed into a single
d_model vector per draw (BERT-style token+segment sum), which keeps the
transformer's sequence length at W+1 draws instead of 9*(W+1) raw tokens.
"""

import os
import json

import torch
import torch.nn as nn

from tokenizer import (
    COLUMN_EMBED_TYPE, TARGET_KNOWN_COLS,
    TYPE_SLOT, TYPE_DOW, TYPE_MACHINE, TYPE_BALLSET, TYPE_DIGIT,
    DIGIT_BASE_SIZE,
)

# ---------------------------------------------------------------------------
# Constants (chosen, plans/PHASE3_PLAN.md section 3 + task instructions)
# ---------------------------------------------------------------------------
NUM_DIGIT_POSITIONS = 3
DIGIT_OUT_SIZE = DIGIT_BASE_SIZE  # 10-way softmax per position

EMBED_DIM = 128          # shared per-token-type embedding dim; == M1 d_model

# M1: decoder-only transformer
M1_D_MODEL = EMBED_DIM
M1_N_LAYERS = 4
M1_N_HEADS = 4
M1_FF_DIM = 512
M1_DROPOUT = 0.1          # chosen, not swept -- not specified by the plan

# M2: GRU
M2_HIDDEN = 256
M2_N_LAYERS = 2

MODEL_M1 = "m1"
MODEL_M2 = "m2"


# ---------------------------------------------------------------------------
# Shared draw embedding
# ---------------------------------------------------------------------------
class DrawEmbedding(nn.Module):
    """One nn.Embedding per token type; sums the columns of a draw (or of
    target_known) into a single embed_dim vector."""

    def __init__(self, vocab, embed_dim):
        super().__init__()
        self.slot_embed = nn.Embedding(vocab["slot"]["size"], embed_dim)
        self.dow_embed = nn.Embedding(vocab["dow"]["size"], embed_dim)
        self.machine_embed = nn.Embedding(vocab["machine"]["size"], embed_dim)
        self.ballset_embed = nn.Embedding(vocab["ballset"]["size"], embed_dim)
        self.digit_embed = nn.Embedding(vocab["digit"]["size"], embed_dim)
        self._table_by_type = {
            TYPE_SLOT: self.slot_embed,
            TYPE_DOW: self.dow_embed,
            TYPE_MACHINE: self.machine_embed,
            TYPE_BALLSET: self.ballset_embed,
            TYPE_DIGIT: self.digit_embed,
        }
        # Precompute the embed-type for each column of target_known, in order.
        self._target_known_types = [COLUMN_EMBED_TYPE[c] for c in TARGET_KNOWN_COLS]

    def _embed_columns(self, tokens, col_types):
        out = None
        for i, t in enumerate(col_types):
            e = self._table_by_type[t](tokens[..., i])
            out = e if out is None else out + e
        return out

    def embed_draws(self, context):
        """context: (B, W, 9) long -> (B, W, embed_dim)."""
        return self._embed_columns(context, COLUMN_EMBED_TYPE)

    def embed_target_known(self, target_known):
        """target_known: (B, 6) long -> (B, embed_dim)."""
        return self._embed_columns(target_known, self._target_known_types)


# ---------------------------------------------------------------------------
# M1: decoder-only transformer
# ---------------------------------------------------------------------------
class M1DecoderTransformer(nn.Module):
    def __init__(self, vocab, W, d_model=M1_D_MODEL, n_layers=M1_N_LAYERS,
                 n_heads=M1_N_HEADS, ff_dim=M1_FF_DIM, dropout=M1_DROPOUT):
        super().__init__()
        self.embed = DrawEmbedding(vocab, d_model)
        self.seq_len = W + 1  # W context draws + 1 target_known position
        self.pos_embed = nn.Embedding(self.seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, DIGIT_OUT_SIZE) for _ in range(NUM_DIGIT_POSITIONS)]
        )
        causal = torch.triu(torch.full((self.seq_len, self.seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, context, target_known):
        ctx = self.embed.embed_draws(context)                       # (B, W, d)
        tgt = self.embed.embed_target_known(target_known).unsqueeze(1)  # (B, 1, d)
        seq = torch.cat([ctx, tgt], dim=1)                           # (B, W+1, d)
        positions = torch.arange(self.seq_len, device=seq.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0)
        out = self.encoder(seq, mask=self.causal_mask)
        final = out[:, -1, :]                                        # (B, d)
        logits = torch.stack([h(final) for h in self.heads], dim=1)   # (B, 3, 10)
        return logits


# ---------------------------------------------------------------------------
# M2: GRU
# ---------------------------------------------------------------------------
class M2GRUModel(nn.Module):
    def __init__(self, vocab, W=None, embed_dim=EMBED_DIM, hidden=M2_HIDDEN, n_layers=M2_N_LAYERS):
        super().__init__()
        self.embed = DrawEmbedding(vocab, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, num_layers=n_layers, batch_first=True)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, DIGIT_OUT_SIZE) for _ in range(NUM_DIGIT_POSITIONS)]
        )

    def forward(self, context, target_known):
        ctx = self.embed.embed_draws(context)
        tgt = self.embed.embed_target_known(target_known).unsqueeze(1)
        seq = torch.cat([ctx, tgt], dim=1)
        out, _ = self.gru(seq)
        final = out[:, -1, :]
        logits = torch.stack([h(final) for h in self.heads], dim=1)
        return logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(model_name, vocab, W, n_layers=None, d_model=None, n_heads=None,
                 ff_dim=None, dropout=None):
    """n_layers/d_model/n_heads/ff_dim/dropout are optional M1 architecture
    overrides (plans/PHASE3_PLAN.md section 3, M3: 2 layers, d_model 128).
    None means "use the M1 default" -- existing M1/M2 configs never set
    these fields, so passing them through changes nothing for M1/M2 runs.
    Ignored for M2 (GRU has no equivalent config surface here)."""
    if model_name == MODEL_M1:
        kwargs = {}
        if n_layers is not None:
            kwargs["n_layers"] = n_layers
        if d_model is not None:
            kwargs["d_model"] = d_model
        if n_heads is not None:
            kwargs["n_heads"] = n_heads
        if ff_dim is not None:
            kwargs["ff_dim"] = ff_dim
        if dropout is not None:
            kwargs["dropout"] = dropout
        return M1DecoderTransformer(vocab, W, **kwargs)
    if model_name == MODEL_M2:
        return M2GRUModel(vocab, W)
    raise ValueError(f"build_model: unknown model_name {model_name!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    _DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    W_DEFAULT = 64
    m1 = M1DecoderTransformer(vocab, W_DEFAULT)
    m2 = M2GRUModel(vocab, W_DEFAULT)
    print(f"M1 decoder-only transformer: {count_params(m1):,} parameters")
    print(f"M2 GRU: {count_params(m2):,} parameters")
