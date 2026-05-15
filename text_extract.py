"""
text_extract.py

For each sample in the test filelist:
  1. Run the CLEAN wav through WavLM + ctc_proj → frame-level char labels.
  2. Run the NOISY wav through WavLM → content embeddings.
  3. For each consecutive run of the same non-blank char label, average the
     corresponding noisy content embeddings and save as:

       {output_dir}/{stem}_{char}_{idx}.npy

     where idx is a per-character occurrence counter (0, 1, 2, ...).

Filelist format (pipe-separated):
  clean_path | noise_path | noisy_path | text | snr
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np
import yaml
import torch
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from wavlm_lora import WavLMLoRASE
from datasets import ID2CHAR, BLANK_ID


# ─────────────────────────────────────────────────────────────────────────────

def build_se_model(ckpt_path: str, device: torch.device, se_cfg: dict = None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "se_cfg" in ckpt:
        se_cfg = ckpt["se_cfg"]
    elif se_cfg is None:
        raise ValueError("Checkpoint has no 'se_cfg' key. Provide --se_cfg.")

    se_model = WavLMLoRASE(se_cfg).to(device)
    result = se_model.load_state_dict(ckpt["se_model"], strict=False)
    if result.missing_keys:
        print(f"[se_model] missing keys: {result.missing_keys}")
    se_model.eval()
    return se_model


@torch.no_grad()
def encode_content(se_model, wav: torch.Tensor, audio_len: torch.Tensor):
    """
    Returns:
        content_emb : (1, T_frames, 1024)
        frame_lens  : (1,)
    """
    T         = wav.shape[1]
    attn_mask = (
        torch.arange(T, device=wav.device).unsqueeze(0) < audio_len.unsqueeze(1)
    ).long()
    out         = se_model.wavlm(wav, attention_mask=attn_mask, output_hidden_states=True)
    content_emb = se_model._extract_content(out.hidden_states)  # (1, T, 1024)
    T_frames    = content_emb.size(1)
    frame_lens  = se_model.frame_lengths(audio_len).clamp(max=T_frames)
    return content_emb, frame_lens


def get_char_segments(ctc_logits: torch.Tensor, frame_len: int, blank_id: int = BLANK_ID):
    """
    ctc_logits : (T, vocab_size)

    Returns list of (char_id, start, end_exclusive) for non-blank runs.
    """
    preds = ctc_logits[:frame_len].argmax(-1).cpu().tolist()

    segments = []
    i = 0
    while i < len(preds):
        curr = preds[i]
        j    = i + 1
        while j < len(preds) and preds[j] == curr:
            j += 1
        if curr != blank_id:
            segments.append((curr, i, j))
        i = j
    return segments


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract per-character content embeddings from noisy speech."
    )
    p.add_argument("-C", "--config",
                   default=str(_HERE / "configs" / "config_stage2.yaml"))
    p.add_argument("--ckpt",     required=True,
                   help="Path to trained checkpoint .tar")
    p.add_argument("--se_cfg",   default=str(_HERE / "config_latent.yaml"),
                   help="Fallback se_model config YAML (overridden by ckpt's se_cfg if present)")
    p.add_argument("--filelist",
                   default="/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt")
    p.add_argument("--output_dir", default="text_emb_dir",
                   help="Directory where .npy files are saved")
    p.add_argument("--max_per_char", type=int, default=10,
                   help="Max embeddings to save per character per folder (sp excluded)")
    p.add_argument("-D", "--device", default="0")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    fs = config["samplerate"]

    se_cfg = None
    if args.se_cfg:
        try:
            with open(args.se_cfg) as f:
                se_cfg = yaml.safe_load(f)
            print(f"[se_cfg] loaded from {args.se_cfg}")
        except FileNotFoundError:
            print(f"[se_cfg] {args.se_cfg} not found; will use ckpt's embedded se_cfg")

    print(f"[ckpt]   {args.ckpt}")
    print(f"[output] {args.output_dir}")

    se_model = build_se_model(args.ckpt, device, se_cfg)
    blank_id = se_model.ctc_loss_fn.blank

    # ── read filelist ─────────────────────────────────────────────────────────
    samples = []
    with open(args.filelist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts      = [p.strip() for p in line.split("|")]
            clean_path = parts[0]
            noisy_path = parts[2]
            snr        = int(float(parts[4])) if len(parts) > 4 else 0
            if "with_reverb" in noisy_path:
                reverb = "with_reverb"
            elif "without_reverb" in noisy_path or "no_reverb" in noisy_path:
                reverb = "no_reverb"
            else:
                reverb = "unknown"
            samples.append((clean_path, noisy_path, reverb, snr))

    print(f"[filelist] {len(samples)} samples from {args.filelist}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # count already-saved embeddings per (folder, char) — supports resuming
    folder_char_counts: dict[tuple, int] = defaultdict(int)
    for existing in out_root.rglob("*_emb.npy"):
        parts = existing.stem.rsplit("_", 3)   # [file_stem, char, idx, 'emb']
        if len(parts) == 4 and re.fullmatch(r"[a-z]|ap", parts[1]):
            folder_char_counts[(existing.parent, parts[1])] += 1

    max_per_char = args.max_per_char

    _CHAR_SAFE = {' ': 'sp', "'": 'ap'}

    # ── main loop ─────────────────────────────────────────────────────────────
    for clean_path, noisy_path, reverb, snr in tqdm(samples, desc="extracting", dynamic_ncols=True):

        out_dir = out_root / reverb / f"snr_{snr}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Clean wav → frame-level char labels
        clean_np  = librosa.load(clean_path, sr=fs, mono=True)[0]
        clean_t   = torch.from_numpy(clean_np).unsqueeze(0).to(device)
        clean_len = torch.tensor([clean_t.shape[1]], dtype=torch.long, device=device)

        content_clean, frame_lens_clean = encode_content(se_model, clean_t, clean_len)
        ctc_logits_clean   = se_model.ctc_proj(content_clean[0]).detach()   # (T, vocab_size)
        valid_frames_clean = int(frame_lens_clean[0].item())

        segments = get_char_segments(ctc_logits_clean, valid_frames_clean, blank_id)

        # filter: drop sp, drop chars already at limit
        valid_segments = [
            (char_id, start, end,
             _CHAR_SAFE.get(ID2CHAR.get(char_id, "?"), ID2CHAR.get(char_id, "?")))
            for char_id, start, end in segments
            if (ch := _CHAR_SAFE.get(ID2CHAR.get(char_id, "?"), ID2CHAR.get(char_id, "?"))) != "sp"
            and folder_char_counts[(out_dir, ch)] < max_per_char
        ]

        if not valid_segments:
            continue

        # 2. Noisy wav → content embeddings and fc outputs (what we actually save)
        noisy_np  = librosa.load(noisy_path, sr=fs, mono=True)[0]
        noisy_t   = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
        noisy_len = torch.tensor([noisy_t.shape[1]], dtype=torch.long, device=device)

        content_noisy, frame_lens_noisy = encode_content(se_model, noisy_t, noisy_len)
        noisy_emb   = content_noisy[0].detach().cpu().numpy()              # (T, 1024)
        noisy_fc    = se_model.ctc_proj(content_noisy[0]).detach().cpu().numpy()  # (T, vocab_size)
        valid_noisy = int(frame_lens_noisy[0].item())

        # 3. Save per-character averaged embeddings
        stem       = re.sub(r"_snr-?\d+", "", Path(noisy_path).stem)
        char_count = defaultdict(int)

        for char_id, start, end, char in valid_segments:
            if folder_char_counts[(out_dir, char)] >= max_per_char:
                continue

            start_n = min(start, valid_noisy)
            end_n   = min(end,   valid_noisy)
            if start_n >= end_n:
                continue

            idx = char_count[char]
            char_count[char] += 1

            np.save(str(out_dir / f"{stem}_{char}_{idx}_emb.npy"),
                    noisy_emb[start_n:end_n].mean(axis=0))   # (1024,)
            np.save(str(out_dir / f"{stem}_{char}_{idx}_fc.npy"),
                    noisy_fc[start_n:end_n].mean(axis=0))    # (vocab_size,)
            folder_char_counts[(out_dir, char)] += 1

    print("Done.")


if __name__ == "__main__":
    main()
