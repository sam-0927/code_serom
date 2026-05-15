"""
Inference script for Stage-2 no-transformer trained model
(WavLM + mel_head + WavLMDec vocoder, no TransformerDecoder).

Reads filelist (format: clean | noise | noisy | text | snr)
and saves enhanced audio organized by reverb condition and SNR:

    <output_dir>/
        with_reverb/
            snr_-10/  stem_enh.wav  stem_clean.wav ...
            snr_0/    ...
        no_reverb/
            snr_-10/  ...
"""

import os
import re
import sys
import argparse
from pathlib import Path

import librosa
import numpy as np
import yaml
import torch
import soundfile as sf
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from wavlm_lora import WavLMLoRASE
from vocoder.wavlmdec import WavLMDec


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_models(ckpt_path: str, config: dict, device: torch.device, se_cfg: dict = None):
    """
    ckpt_path : .tar saved by train_stage2_no_trans.py
                (must contain 'se_model' and 'generator' keys)
    config    : training config yaml (config_stage2.yaml)
    se_cfg    : fallback se_model config; overridden by ckpt's 'se_cfg' if present
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "se_cfg" in ckpt:
        se_cfg = ckpt["se_cfg"]
    elif se_cfg is None:
        raise ValueError("Checkpoint has no 'se_cfg' key. Please provide --se_cfg.")

    se_cfg["n_mels"] = config.get("n_mels", 80)

    # ── se_model (includes mel_head) ──────────────────────────────────────────
    se_model = WavLMLoRASE(se_cfg).to(device)
    result = se_model.load_state_dict(ckpt["se_model"], strict=False)
    if result.missing_keys:
        print(f"[se_model] missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[se_model] unexpected keys (ignored): {result.unexpected_keys}")
    se_model.eval()
    print(f"[se_model]  loaded from {ckpt_path}")

    # ── generator (vocoder) ───────────────────────────────────────────────────
    voc_cfg = {k: v for k, v in config["vocoder_config"].items()
               if k not in ("cond_mode",)}
    generator = WavLMDec(
        cond_dim  = None,
        cond_mode = "concat",
        spk_dim   = None,
        **voc_cfg,
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print(f"[generator] loaded from {ckpt_path}")

    return se_model, generator


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_sample(se_model, generator, noisy_path: str,
               device: torch.device, fs: int = 16000, cond: str = "both"):
    wav_np = librosa.load(noisy_path, sr=fs, mono=True)[0]    # (T,)
    wav    = torch.from_numpy(wav_np).unsqueeze(0).to(device)  # (1, T)
    alen   = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)

    # ── 1. WavLM encode ───────────────────────────────────────────────────────
    T         = wav.shape[1]
    attn_mask = (
        torch.arange(T, device=device).unsqueeze(0) < alen.unsqueeze(1)
    ).long()
    out    = se_model.wavlm(wav, attention_mask=attn_mask, output_hidden_states=True)
    hidden = out.hidden_states

    content_emb  = hidden[se_model.content_layer]   # (1, T_frames, 1024)
    acoustic_emb = hidden[se_model.acoustic_layer]  # (1, T_frames, 1024)

    # ── 2. Build mel input (mirrors _frozen_forward in train_stage2_no_trans.py)
    content_fc = se_model.ctc_proj(content_emb)
    acoustic_h = se_model.proj_acoustic(acoustic_emb)
    if cond in ("content", "both"):
        content_h  = se_model.proj_content(content_fc)
        input_feat = acoustic_h + content_h
    else:
        input_feat = acoustic_h

    # ── 3. mel_head → mel ─────────────────────────────────────────────────────
    mel_pred = se_model.mel_head(input_feat)   # (1, T_frames, n_mels)

    # ── 4. Vocoder ────────────────────────────────────────────────────────────
    enh_wav = generator(mel_pred)  # (1, T_samples)

    return enh_wav[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference (no-transformer): noisy → enhanced speech, saved by reverb/SNR"
    )
    p.add_argument("-C", "--config",
                   default=str(_HERE / "configs" / "config_stage2.yaml"))
    p.add_argument("--ckpt",    required=True,
                   help="Path to stage-2 no-trans checkpoint .tar")
    p.add_argument("--se_cfg",  default=None,
                   help="Path to se_model config yaml (fallback if ckpt has no se_cfg key)")
    p.add_argument("--filelist",
                   default="/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt")
    p.add_argument("--output_dir", default="no_trans",
                   help="Root directory for enhanced outputs")
    p.add_argument("-D", "--device", default="0",
                   help="CUDA device index")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fs   = config["samplerate"]
    cond = config.get("cond", "both")

    print(f"[ckpt]   {args.ckpt}")
    print(f"[cond]   {cond}")
    print(f"[output] {args.output_dir}")

    # ── load se_cfg fallback ──────────────────────────────────────────────────
    se_cfg = None
    if args.se_cfg:
        try:
            with open(args.se_cfg) as f:
                se_cfg = yaml.safe_load(f)
            print(f"[se_cfg] loaded from {args.se_cfg} (overridden by ckpt if present)")
        except FileNotFoundError:
            print(f"[se_cfg] {args.se_cfg} not found; will use ckpt's embedded se_cfg")

    # ── load models ───────────────────────────────────────────────────────────
    se_model, generator = build_models(args.ckpt, config, device, se_cfg=se_cfg)

    # ── read filelist ─────────────────────────────────────────────────────────
    samples = []
    with open(args.filelist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            clean_path = parts[0]
            noisy_path = parts[2]
            snr        = int(float(parts[4])) if len(parts) > 4 else 0
            if "with_reverb" in noisy_path:
                reverb = "with_reverb"
            elif ("without_reverb" in noisy_path) or ("no_reverb" in noisy_path):
                reverb = "no_reverb"
            else:
                reverb = "unknown"
            samples.append((clean_path, noisy_path, reverb, snr))

    print(f"[filelist] {len(samples)} samples from {args.filelist}")

    # ── run inference ─────────────────────────────────────────────────────────
    for clean_path, noisy_path, reverb, snr in tqdm(samples, desc="inference", dynamic_ncols=True):
        enh = run_sample(
            se_model, generator,
            noisy_path=noisy_path,
            device=device,
            fs=fs,
            cond=cond,
        )

        snr_str = f"snr_{snr}"
        out_dir = Path(args.output_dir) / reverb / snr_str
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = re.sub(r'_snr-?\d+', '', Path(noisy_path).stem)
        sf.write(str(out_dir / f"{stem}_enh.wav"), enh, fs)

        clean_wav = librosa.load(clean_path, sr=fs, mono=True)[0]
        sf.write(str(out_dir / f"{stem}_clean.wav"), clean_wav, fs)

    print("Done.")


if __name__ == "__main__":
    main()
