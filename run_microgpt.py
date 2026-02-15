"""
Multi-dataset GPT trainer.
Wraps the GPU/PyTorch atomic GPT into a single train() function and
runs it across all suggested datasets automatically.

Folder layout produced:
  datasets/          <- cached raw text files (never re-downloaded)
  outputs/<name>/    <- per-dataset: loss.png, ckpt.pt, samples.txt

Usage:
  python run_microgpt.py                        # train all datasets
  python run_microgpt.py --only pokemon names   # train a subset
  python run_microgpt.py --steps 500            # quick smoke-test all

@karpathy (original), multi-dataset runner
"""

import os
import sys
import math
import random
import argparse
import json
import urllib.request
import urllib.error

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg' if 'DISPLAY' in __import__('os').environ or __import__('sys').platform == 'darwin' else 'Agg')
import matplotlib.pyplot as plt
plt.ion()   # interactive mode -- enables live plot updates during training

# ---------------------------------------------------------------------------
# Dataset registry
# Each entry is passed as **kwargs to train(), so any train() param can be
# overridden per dataset (steps, n_embd, n_layer, temperature, topk, etc.)
# ---------------------------------------------------------------------------
DATASETS = [
    dict(
        name        = "pokemon",
        url         = "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv",
        csv_col     = "identifier",   # extract one column from CSV
        steps       = 1000,
        n_embd      = 32,
        n_layer     = 2,
        block_size  = 12,
        temperature = 0.6,
        topk        = 5,
        note        = "Pokemon names -- short sequences, fast to learn",
    ),
    dict(
        name        = "paul_graham",
        # url is the GitHub API endpoint for the text_data folder
        # the loader fetches the file list then downloads each .txt essay
        url         = "https://api.github.com/repos/sgoel97/essay-datasets/contents/paul_graham_essays/text_data",
        loader      = "pg_essays",    # triggers load_pg_essays() instead of load_docs_from_file()
        stream_mode = True,            # inference: continuous text, don't stop at BOS
        steps       = 5000,
        n_embd      = 64,
        n_head      = 4,
        n_layer     = 4,
        block_size  = 128,
        batch_size  = 4,
        temperature = 0.8,
        topk        = 10,
        note        = "Paul Graham essays (~200 txt files) -- distinctive prose style",
    ),
    dict(
        name        = "cities",
        url         = "https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv",
        csv_col     = "name",
        steps       = 3000,
        n_embd      = 32,
        n_layer     = 2,
        block_size  = 16,
        temperature = 0.7,
        topk        = 5,
        note        = "World city names -- more diversity, longer sequences",
    ),
    dict(
        name        = "names",
        url         = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt",
        steps       = 2000,
        n_embd      = 16,
        n_layer     = 1,
        block_size  = 8,
        temperature = 0.6,
        note        = "Baby names -- the original benchmark",
    ),
    dict(
        name        = "english_words",
        url         = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
        steps       = 3000,
        n_embd      = 32,
        n_layer     = 2,
        block_size  = 16,
        temperature = 0.7,
        topk        = 5,
        note        = "370k English words -- morphological patterns",
    ),
    dict(
        name        = "shakespeare",
        url         = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        steps       = 5000,
        n_embd      = 64,
        n_head      = 4,
        n_layer     = 4,
        block_size  = 64,
        batch_size  = 4,
        temperature = 0.8,
        topk        = 10,
        stream_mode = True,            # inference: continuous text, don't stop at BOS
        note        = "Tiny Shakespeare -- char-level language model classic",
    ),
]

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATASETS_DIR = "datasets"
OUTPUTS_DIR  = "outputs"
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR,  exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset download + parsing helpers
# ---------------------------------------------------------------------------
def download(url: str, dest: str):
    """Download url -> dest, skip if already cached."""
    if os.path.exists(dest):
        print(f"  [cache] {dest}")
        return
    print(f"  [download] {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [saved]  {dest}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

def load_docs_from_file(path: str, csv_col: str = None) -> list[str]:
    """
    Load a text file as a list of documents (one per line).
    If csv_col is set, treat the file as a CSV and extract that column,
    normalising to lowercase and stripping non-alpha characters so the
    tokenizer stays small and comparable across datasets.
    """
    with open(path, encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    if csv_col:
        lines  = raw.strip().split('\n')
        # Find column index from header
        header = [h.strip().strip('"').lower() for h in lines[0].split(',')]
        if csv_col.lower() not in header:
            raise ValueError(f"Column '{csv_col}' not found in {path}. Available: {header}")
        col_idx = header.index(csv_col.lower())
        docs = []
        for line in lines[1:]:
            parts = line.split(',')
            if col_idx < len(parts):
                val = parts[col_idx].strip().strip('"').lower()
                # Keep only alphabetic + spaces/hyphens, drop empty
                val = ''.join(c for c in val if c.isalpha() or c in (' ', '-'))
                if val:
                    docs.append(val)
    else:
        docs = [l.strip() for l in raw.strip().split('\n') if l.strip()]

    return docs

def load_pg_essays(api_url: str, cache_dir: str) -> list[str]:
    """
    Fetch all Paul Graham .txt essays from the sgoel97 GitHub repo.
    Uses the GitHub Contents API to get the file list, then downloads
    each .txt file individually. All files are cached under cache_dir.
    Returns one document per paragraph (non-empty lines) across all essays.
    """
    index_path = os.path.join(cache_dir, "_pg_index.json")

    # Fetch the folder listing from GitHub API (cached after first run)
    if not os.path.exists(index_path):
        print(f"  [download] GitHub file listing: {api_url}")
        req = urllib.request.Request(api_url,
              headers={"User-Agent": "Mozilla/5.0",
                       "Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req) as r:
            file_list = json.loads(r.read().decode())
        with open(index_path, 'w') as f:
            json.dump(file_list, f)
        print(f"  [saved]    {index_path}  ({len(file_list)} files)")
    else:
        with open(index_path) as f:
            file_list = json.load(f)
        print(f"  [cache]    {index_path}  ({len(file_list)} files)")

    docs = []
    for entry in file_list:
        if not entry.get("name", "").endswith(".txt"):
            continue
        fname  = entry["name"]
        fpath  = os.path.join(cache_dir, fname)
        raw_url = entry.get("download_url", "")

        if not os.path.exists(fpath):
            if not raw_url:
                continue
            try:
                urllib.request.urlretrieve(raw_url, fpath)
            except Exception as e:
                print(f"  [warn] could not download {fname}: {e}")
                continue

        with open(fpath, encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Each non-empty line becomes a document (paragraph-level training)
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 20:   # skip very short lines / headers
                docs.append(line)

    print(f"  loaded {len(docs)} paragraphs from {len(file_list)} PG essays")
    return docs


# ---------------------------------------------------------------------------
# Model definition (identical to gpt_gpu.py)
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
        H, D    = self.n_head, C // self.n_head
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

# ---------------------------------------------------------------------------
# Core train() function
# Every hyperparameter has a sane default; pass overrides per dataset.
# ---------------------------------------------------------------------------
def train(
    name        : str,
    url         : str,
    # model
    n_embd      : int   = 16,
    n_head      : int   = 4,
    n_layer     : int   = 1,
    block_size  : int   = 8,
    # training
    steps       : int   = 2000,
    lr          : float = 1e-2,
    batch_size  : int   = 1,
    clip_grad   : float = 1.0,
    val_every   : int   = 100,
    val_docs    : int   = 20,
    seed        : int   = 42,
    # inference
    temperature : float = 0.6,
    topk        : int   = 0,
    n_samples   : int   = 20,
    # data
    csv_col     : str   = None,   # if set, parse this column from a CSV
    loader      : str   = None,   # if set, use a custom loader instead of load_docs_from_file
    stream_mode : bool  = False,  # if True, generate continuous text instead of stopping at BOS
    # ignored meta fields
    note        : str   = "",
    **_,                           # absorb any unrecognised kwargs silently
):
    out_dir = os.path.join(OUTPUTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path    = os.path.join(out_dir, "ckpt.pt")
    plot_path    = os.path.join(out_dir, "loss.png")
    samples_path = os.path.join(out_dir, "samples.txt")
    data_path    = os.path.join(DATASETS_DIR, f"{name}.txt")

    print(f"\n{'='*60}")
    print(f"  Dataset : {name}")
    print(f"  Note    : {note}")
    print(f"  Steps   : {steps}  |  n_embd={n_embd}  n_layer={n_layer}  n_head={n_head}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # -----------------------------------------------------------------------
    # Download + load dataset
    # Dispatch to a custom loader if specified, otherwise use the default.
    # -----------------------------------------------------------------------
    if loader == "pg_essays":
        # PG essays: multi-file download from GitHub, cached per-file
        pg_cache_dir = os.path.join(DATASETS_DIR, "paul_graham_essays")
        os.makedirs(pg_cache_dir, exist_ok=True)
        docs = load_pg_essays(url, pg_cache_dir)
    else:
        # Default: single-file download + optional CSV column extraction
        download(url, data_path)
        docs = load_docs_from_file(data_path, csv_col=csv_col)

    random.shuffle(docs)
    print(f"num docs: {len(docs)}")
    if not docs:
        print(f"  [SKIP] No documents loaded for {name}, skipping.")
        return

    # CHANGE 3. Train / val split (90/10)
    split_idx  = int(0.9 * len(docs))
    train_docs = docs[:split_idx]
    val_docs_  = docs[split_idx:]
    print(f"train: {len(train_docs)}  |  val: {len(val_docs_)}")

    # Build vocab
    chars      = ["<BOS>"] + sorted(set("".join(docs)))
    stoi       = {ch: i for i, ch in enumerate(chars)}
    itos       = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    bos_id     = stoi["<BOS>"]
    train_docs = [d for d in train_docs if all(ch in stoi for ch in d)]
    val_docs_  = [d for d in val_docs_  if all(ch in stoi for ch in d)]
    print(f"vocab size: {vocab_size}")

    # -----------------------------------------------------------------------
    # Model + optimizer
    # -----------------------------------------------------------------------
    model     = GPT(vocab_size, n_embd=n_embd, n_head=n_head,
                    n_layer=n_layer, block_size=block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    print(f"num params: {sum(p.numel() for p in model.parameters()):,}")

    def encode_doc(doc):
        tokens = [bos_id] + [stoi[ch] for ch in doc] + [bos_id]
        n      = min(block_size, len(tokens) - 1)
        return tokens[:n], tokens[1:n + 1]

    # CHANGE 3. Validation loss helper
    def compute_val_loss():
        sample = random.sample(val_docs_, min(val_docs, len(val_docs_)))
        model.eval()
        total = 0.0
        with torch.no_grad():
            for doc in sample:
                x_ids, y_ids = encode_doc(doc)
                x = torch.tensor([x_ids], dtype=torch.long, device=device)
                y = torch.tensor([y_ids], dtype=torch.long, device=device)
                logits = model(x)
                total += F.cross_entropy(logits.view(-1, vocab_size),
                                         y.view(-1), reduction="mean").item()
        model.train()
        return total / len(sample)

    # CHANGE 6. Top-k filter
    def top_k_filter(probs: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or k >= probs.size(-1):
            return probs
        topk_vals, _ = torch.topk(probs, k)
        probs = probs.masked_fill(probs < topk_vals[..., -1, None], 0.0)
        return probs / probs.sum(dim=-1, keepdim=True)

    # CHANGE 7. generate(prompt)
    # stream_mode=False (default): stop at BOS -- good for names/words/compounds
    # stream_mode=True:            ignore BOS, generate up to max_tokens chars -- good for prose
    def generate(prompt='', temp=None, k=None, max_tokens=None):
        temp       = temp      or temperature
        k          = k         if k is not None else topk
        max_tokens = max_tokens or (block_size * 4 if stream_mode else block_size)

        model.eval()
        token_ids = [bos_id] + [stoi[ch] for ch in prompt if ch in stoi]
        with torch.no_grad():
            for _ in range(max_tokens):
                ctx    = token_ids[-block_size:]
                x      = torch.tensor([ctx], dtype=torch.long, device=device)
                logits = model(x)
                probs  = F.softmax(logits[0, -1] / temp, dim=-1)
                probs  = top_k_filter(probs, k)
                nxt    = torch.multinomial(probs, num_samples=1).item()
                if nxt == bos_id:
                    if stream_mode:
                        continue   # treat BOS as a paragraph break, keep going
                    else:
                        break      # BOS = end of word/name, stop here
                token_ids.append(nxt)
        model.train()
        gen_ids = token_ids[1 + len([ch for ch in prompt if ch in stoi]):]
        return prompt + ''.join(itos[i] for i in gen_ids)

    # -----------------------------------------------------------------------
    # CHANGE 2. EMA + CHANGE 8. loss history + live plot setup
    # -----------------------------------------------------------------------
    ema_loss           = None
    ema_alpha          = 0.95
    train_loss_history = []
    val_loss_history   = []

    # Open a live figure for this dataset -- updates every plot_every steps
    plot_every = max(1, steps // 200)   # ~200 redraws per run, regardless of step count
    fig_live, ax_live = plt.subplots(figsize=(9, 4))
    fig_live.patch.set_facecolor('#0d1117')
    ax_live.set_facecolor('#0d1117')
    ax_live.set_xlabel('step', color='#8b949e')
    ax_live.set_ylabel('loss', color='#8b949e')
    ax_live.tick_params(colors='#8b949e')
    for spine in ax_live.spines.values():
        spine.set_edgecolor('#30363d')
    ax_live.grid(color='#21262d', linewidth=0.5)
    fig_live.suptitle(f'training: {name}', color='#e6edf3', fontsize=11)
    plt.tight_layout()
    plt.show(block=False)

    def redraw_live():
        ax_live.cla()
        ax_live.set_facecolor('#0d1117')
        ax_live.set_xlabel('step', color='#8b949e')
        ax_live.set_ylabel('loss', color='#8b949e')
        ax_live.tick_params(colors='#8b949e')
        for spine in ax_live.spines.values():
            spine.set_edgecolor('#30363d')
        ax_live.grid(color='#21262d', linewidth=0.5)

        s_tr  = [s for s, _, __ in train_loss_history]
        r_tr  = [r for _, r, __ in train_loss_history]
        e_tr  = [e for _, __, e in train_loss_history]
        s_val = [s for s, _ in val_loss_history]
        v_val = [v for _, v in val_loss_history]

        ax_live.plot(s_tr, r_tr, color='#1f6feb', linewidth=0.7,
                     alpha=0.4, label='train (raw)')
        ax_live.plot(s_tr, e_tr, color='#58a6ff', linewidth=2.0,
                     label='train (ema)')
        if s_val:
            ax_live.plot(s_val, v_val, color='#f0883e', linewidth=1.5,
                         marker='o', markersize=4, label='val')

        pct  = 100 * len(train_loss_history) / steps
        cur_ema = e_tr[-1] if e_tr else 0
        cur_val = f'{v_val[-1]:.4f}' if v_val else 'n/a'
        ax_live.set_title(
            f'{name}  |  step {len(train_loss_history)}/{steps} ({pct:.0f}%)'
            f'  ema {cur_ema:.4f}  val {cur_val}',
            color='#e6edf3', fontsize=9, pad=6
        )
        ax_live.legend(facecolor='#161b22', edgecolor='#30363d',
                       labelcolor='#e6edf3', fontsize=8)
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    for step in range(steps):
        losses = []
        for b in range(max(1, batch_size)):
            doc    = train_docs[(step * batch_size + b) % len(train_docs)]
            x_ids, y_ids = encode_doc(doc)
            x      = torch.tensor([x_ids], dtype=torch.long, device=device)
            y      = torch.tensor([y_ids], dtype=torch.long, device=device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.view(-1, vocab_size),
                                          y.view(-1), reduction="mean"))

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # CHANGE 1. Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Linear LR decay
        lr_t = lr * (1 - step / max(1, steps - 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t

        optimizer.step()

        # CHANGE 2. Record EMA loss
        raw      = loss.item()
        ema_loss = raw if ema_loss is None else ema_alpha * ema_loss + (1 - ema_alpha) * raw
        train_loss_history.append((step + 1, raw, ema_loss))

        # CHANGE 3. Periodic val loss
        if (step + 1) % val_every == 0:
            vl = compute_val_loss()
            val_loss_history.append((step + 1, vl))

        # Print every n/10 steps so you get exactly 10 log lines per dataset
        print_every = max(1, steps // 10)
        if (step + 1) % print_every == 0 or step == steps - 1:
            cur_val = f'{val_loss_history[-1][1]:.4f}' if val_loss_history else 'n/a'
            print(f"  [{name}] step {step+1:4d}/{steps} ({100*(step+1)//steps:3d}%)"
                  f"  loss {raw:.4f}  ema {ema_loss:.4f}  val {cur_val}")

        # Live plot update every plot_every steps and on the final step
        if (step + 1) % plot_every == 0 or step == steps - 1:
            redraw_live()

        # CHANGE 5. Checkpoint
        if (step + 1) % val_every == 0:
            torch.save({
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "chars":     chars,
                "stoi":      stoi,
                "itos":      itos,
                "model_cfg": dict(n_embd=n_embd, n_head=n_head,
                                  n_layer=n_layer, block_size=block_size),
            }, ckpt_path)

    plt.close(fig_live)   # close the live window before opening the next dataset

    # -----------------------------------------------------------------------
    # CHANGE 8. Matplotlib plot -> outputs/<name>/loss.png
    # -----------------------------------------------------------------------
    steps_tr  = [s for s, _, __ in train_loss_history]
    raw_tr    = [r for _, r, __ in train_loss_history]
    ema_tr    = [e for _, __, e in train_loss_history]
    steps_val = [s for s, _ in val_loss_history]
    vals_val  = [v for _, v in val_loss_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.plot(steps_tr, raw_tr, color='#1f6feb', linewidth=0.8, alpha=0.5, label='train (raw)')
    ax.plot(steps_tr, ema_tr, color='#58a6ff', linewidth=2.0,             label='train (ema)')
    if steps_val:
        ax.plot(steps_val, vals_val, color='#f0883e', linewidth=1.5,
                marker='o', markersize=5,                                  label='val')
    ax.set_xlabel('step', color='#8b949e')
    ax.set_ylabel('loss', color='#8b949e')
    ax.set_title(f'GPT loss — {name}', color='#e6edf3', pad=12)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
    ax.grid(color='#21262d', linewidth=0.5)
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  loss plot  -> {plot_path}")


    # -----------------------------------------------------------------------
    # CHANGE 7. Inference -- save samples to outputs/<n>/samples.txt
    # stream_mode: one long continuous generation (shakespeare, paul_graham)
    # discrete mode: n_samples short independent samples (names, words, etc.)
    # -----------------------------------------------------------------------
    model.eval()
    lines = [f"=== {name} samples ===", ""]

    if stream_mode:
        # Continuous text -- generate one long stream per seed prompt
        lines.append("--- continuous generation ---")
        seeds = ["", "The ", "I "]   # seed prompts for prose generation
        for seed in seeds:
            out   = generate(prompt=seed)
            label = repr(seed)
            lines += [f"  seed {label}:", f"  {out}", ""]
            print(f"  seed {label}:")
            print(f"{out}")
            print()
    else:
        # Discrete samples -- each is an independent short generation
        lines.append("--- unconditional samples ---")
        for i in range(n_samples):
            out = generate()
            lines.append(f"  {i+1:2d}: {out}")
            print(f"  sample {i+1:2d}: {out}")

        # Seeded samples -- first 3 non-BOS vocab chars as seeds
        seeds = [v for v in list(stoi.keys()) if v != "<BOS>"][:3]
        lines.append("")
        lines.append("--- seeded samples ---")
        for seed in seeds:
            out = generate(prompt=seed)
            lines.append(f"  seed '{seed}' -> {out}")
            print(f"  seed '{seed}' -> {out}")



    with open(samples_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  samples    -> {samples_path}")
    print(f"  checkpoint -> {ckpt_path}")

    return dict(
        name               = name,
        final_ema          = ema_tr[-1],
        final_val          = vals_val[-1] if vals_val else None,
        train_loss_history = train_loss_history,   # for combined plot
        val_loss_history   = val_loss_history,     # for combined plot
    )

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT training on all datasets")
    parser.add_argument("--only",  nargs="*", help="Train only these dataset names")
    parser.add_argument("--steps", type=int,  help="Override steps for all datasets (useful for smoke-testing)")
    args = parser.parse_args()

    datasets_to_run = DATASETS
    if args.only:
        datasets_to_run = [d for d in DATASETS if d["name"] in args.only]
        if not datasets_to_run:
            print(f"No matching datasets found. Available: {[d['name'] for d in DATASETS]}")
            sys.exit(1)

    if args.steps:
        datasets_to_run = [{**d, "steps": args.steps} for d in datasets_to_run]

    print(f"Running {len(datasets_to_run)} dataset(s): {[d['name'] for d in datasets_to_run]}")

    results = []
    for ds in datasets_to_run:
        try:
            result = train(**ds)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n[ERROR] {ds['name']} failed: {e}")
            import traceback; traceback.print_exc()
            continue

    # -----------------------------------------------------------------------
    # Summary table across all datasets
    # -----------------------------------------------------------------------
    if results:
        print(f"\n{'='*50}")
        print(f"  SUMMARY")
        print(f"{'='*50}")
        print(f"  {'dataset':<20} {'final ema loss':>14} {'final val loss':>14}")
        print(f"  {'-'*48}")
        for r in results:
            val_str = f"{r['final_val']:.4f}" if r['final_val'] is not None else "  n/a  "
            print(f"  {r['name']:<20} {r['final_ema']:>14.4f} {val_str:>14}")
        print(f"{'='*50}")
        print(f"\nAll outputs saved under: {os.path.abspath(OUTPUTS_DIR)}/")

    # -----------------------------------------------------------------------
    # Combined loss figure -- one subplot per dataset, all in one PNG
    # Saved to outputs/all_losses.png
    # -----------------------------------------------------------------------
    if results:
        n      = len(results)
        ncols  = min(3, n)                          # max 3 columns
        nrows  = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 4 * nrows),
                                 squeeze=False)
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle('GPT training loss — all datasets',
                     color='#e6edf3', fontsize=13, y=1.01)

        for idx, r in enumerate(results):
            ax  = axes[idx // ncols][idx % ncols]
            ax.set_facecolor('#0d1117')

            tlh = r['train_loss_history']
            vlh = r['val_loss_history']

            steps_tr  = [s for s, _, __ in tlh]
            raw_tr    = [v for _, v, __ in tlh]
            ema_tr    = [e for _, __, e in tlh]
            steps_val = [s for s, _ in vlh]
            vals_val  = [v for _, v in vlh]

            ax.plot(steps_tr, raw_tr, color='#1f6feb',
                    linewidth=0.7, alpha=0.4, label='train (raw)')
            ax.plot(steps_tr, ema_tr, color='#58a6ff',
                    linewidth=1.8,             label='train (ema)')
            if steps_val:
                ax.plot(steps_val, vals_val, color='#f0883e',
                        linewidth=1.4, marker='o', markersize=4, label='val')

            ax.set_title(r['name'], color='#e6edf3', fontsize=11, pad=8)
            ax.set_xlabel('step', color='#8b949e', fontsize=9)
            ax.set_ylabel('loss', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
            ax.grid(color='#21262d', linewidth=0.4)
            ax.legend(facecolor='#161b22', edgecolor='#30363d',
                      labelcolor='#e6edf3', fontsize=8)

        # Hide any unused subplot slots
        for idx in range(len(results), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        combined_path = os.path.join(OUTPUTS_DIR, 'all_losses.png')
        plt.savefig(combined_path, dpi=150,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f"  combined plot -> {combined_path}")