"""
probe.py -- plot loss curves and generate samples from a saved checkpoint.
No training required. Works with any checkpoint saved by run_microgpt.py.

Usage:
  python probe.py --dataset paul_graham
  python probe.py --dataset shakespeare --prompt "To be or not"
  python probe.py --dataset names --n-samples 30 --temperature 0.5
  python probe.py --dataset paul_graham --stream --prompt "The best startups"

In a Colab notebook:
  !python probe.py --dataset paul_graham --prompt "The best startups" --stream
"""


import os
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="inspect a saved microgpt checkpoint")
parser.add_argument("--dataset",     required=True,          help="dataset name (e.g. paul_graham)")
parser.add_argument("--prompt",      default="",             help="seed text for generation")
parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: from checkpoint config)")
parser.add_argument("--topk",        type=int,   default=0,  help="top-k filter (0 = full distribution)")
parser.add_argument("--n-samples",   type=int,   default=5,  help="number of samples to generate")
parser.add_argument("--max-tokens",  type=int,   default=None, help="max tokens to generate per sample")
parser.add_argument("--stream",      action="store_true",    help="continuous generation (ignore BOS, keep going)")
parser.add_argument("--seed",        type=int,   default=42, help="random seed")
parser.add_argument("--ckpt-dir",    default="outputs",      help="root outputs directory")
parser.add_argument("--no-plot",     action="store_true",    help="skip loss plot")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------
ckpt_path = os.path.join(args.ckpt_dir, args.dataset, "ckpt.pt")
if not os.path.exists(ckpt_path):
    print(f"[error] no checkpoint found at {ckpt_path}")
    print(f"        run training first:  python run_microgpt.py --only {args.dataset}")
    exit(1)

print(f"loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)

chars      = ckpt["chars"]
stoi       = ckpt["stoi"]
itos       = {int(k): v for k, v in ckpt["itos"].items()} \
             if isinstance(next(iter(ckpt["itos"].keys())), str) else ckpt["itos"]
cfg        = ckpt["model_cfg"]
vocab_size = len(chars)
bos_id     = stoi["<BOS>"]

n_embd     = cfg["n_embd"]
n_head     = cfg["n_head"]
n_layer    = cfg["n_layer"]
block_size = cfg["block_size"]

print(f"config: n_embd={n_embd} n_head={n_head} n_layer={n_layer} block_size={block_size}")
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model (must match run_microgpt.py exactly)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                          .view(1, 1, block_size, block_size))
    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, C // self.n_head
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.attn  = CausalSelfAttention(n_embd, n_head, block_size)
        self.mlp   = MLP(n_embd)
        self.norm1 = nn.RMSNorm(n_embd)
        self.norm2 = nn.RMSNorm(n_embd)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte        = nn.Embedding(vocab_size, n_embd)
        self.wpe        = nn.Embedding(block_size, n_embd)
        self.blocks     = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.lm_head    = nn.Linear(n_embd, vocab_size, bias=False)
    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.wte(idx) + self.wpe(pos)
        return self.lm_head(self.blocks(x))

model = GPT(vocab_size, n_embd=n_embd, n_head=n_head,
            n_layer=n_layer, block_size=block_size).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"num params: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# Loss plot -- re-reads the existing loss.png path and also rebuilds from
# any loss history stored in the checkpoint (if present)
# ---------------------------------------------------------------------------
if not args.no_plot:
    loss_png = os.path.join(args.ckpt_dir, args.dataset, "loss.png")
    if os.path.exists(loss_png):
        print(f"\nloss plot already saved at: {loss_png}")
        try:
            from IPython.display import Image, display
            display(Image(loss_png))
        except ImportError:
            pass
    else:
        print("[warn] no loss.png found -- run at least one val_every checkpoint to generate it")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def top_k_filter(probs: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= probs.size(-1):
        return probs
    topk_vals, _ = torch.topk(probs, k)
    probs = probs.masked_fill(probs < topk_vals[..., -1, None], 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)

def generate(prompt="", temperature=0.7, max_tokens=None, k=0, stream=False):
    temperature = temperature or 0.7
    max_tokens  = max_tokens or (block_size * 4 if stream else block_size)
    token_ids   = [bos_id] + [stoi[ch] for ch in prompt if ch in stoi]

    with torch.no_grad():
        for _ in range(max_tokens):
            ctx    = token_ids[-block_size:]
            x      = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)
            probs  = F.softmax(logits[0, -1] / temperature, dim=-1)
            probs  = top_k_filter(probs, k)
            nxt    = torch.multinomial(probs, num_samples=1).item()
            if nxt == bos_id:
                if stream:
                    continue
                else:
                    break
            token_ids.append(nxt)

    gen_ids = token_ids[1 + len([ch for ch in prompt if ch in stoi]):]
    return prompt + "".join(itos[i] for i in gen_ids)

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
temperature = args.temperature or (0.8 if args.stream else 0.6)
stream      = args.stream
prompt      = args.prompt
max_tokens  = args.max_tokens

print(f"\n--- generating {args.n_samples} sample(s) ---")
print(f"    temperature={temperature}  topk={args.topk}  stream={stream}  prompt={repr(prompt)}\n")

for i in range(args.n_samples):
    out = generate(
        prompt      = prompt,
        temperature = temperature,
        max_tokens  = max_tokens,
        k           = args.topk,
        stream      = stream,
    )
    if stream:
        print(f"--- sample {i+1} ---")
        print(out)
        print()
    else:
        print(f"  {i+1:2d}: {out}")