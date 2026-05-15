"""
Inference script for Stage-2 (no CTC) trained model
(WavLM + proj_content (no ctc_proj) + TransformerDecoder + WavLMDec vocoder).

Reads filelists/test.txt (format: clean | noise | noisy | text | snr)
and saves enhanced audio organized by reverb condition and SNR:

    <output_dir>/
        with_reverb/
            snr_-10/  stem.wav ...
            snr_0/    ...
        without_reverb/
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
from conformer_decoder import TransformerDecoder
from vocoder.wavlmdec import WavLMDec


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_models(ckpt_path: str, config: dict, device: torch.device, se_cfg: dict = None):
    """
    ckpt_path : path to .tar saved by Stage2Trainer._save_checkpoint
    config    : main training config (config_stage2.yaml)
    se_cfg    : fallback se_model config dict (used only if checkpoint lacks "se_cfg" key)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "se_cfg" in ckpt:
        se_cfg = ckpt["se_cfg"]
    elif se_cfg is None:
        raise ValueError("Checkpoint has no 'se_cfg' key. Please provide --se_cfg.")

    # ── se_model ──────────────────────────────────────────────────────────────
    se_model = WavLMLoRASE(se_cfg).to(device)
    result = se_model.load_state_dict(ckpt["se_model"], strict=False)
    if result.missing_keys:
        print(f"[se_model] missing keys (randomly initialized): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[se_model] unexpected keys (ignored): {result.unexpected_keys}")
    se_model.eval()

    # ── decoder ───────────────────────────────────────────────────────────────
    dec_cfg = se_cfg["decoder"]
    n_mels  = config.get("n_mels", 80)
    decoder = TransformerDecoder(
        input_dim = dec_cfg["d_model"],
        d_model   = dec_cfg["d_model"],
        n_layers  = dec_cfg["n_layers"],
        n_heads   = dec_cfg["n_heads"],
        ffn_dim   = dec_cfg["ffn_dim"],
        dropout   = dec_cfg["dropout"],
        max_len   = dec_cfg.get("max_len", 4096),
        n_mels    = n_mels,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    # ── generator (vocoder) ───────────────────────────────────────────────────
    voc_cfg   = {k: v for k, v in config["vocoder_config"].items()
                 if k not in ("cond_mode",)}
    generator = WavLMDec(
        cond_dim  = None,
        cond_mode = "concat",
        spk_dim   = None,
        **voc_cfg,
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    return se_model, decoder, generator


# ─────────────────────────────────────────────────────────────────────────────
# Encode helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode(se_model, wav: torch.Tensor, audio_len: torch.Tensor):
    """
    wav       : (1, T)  noisy waveform on device
    audio_len : (1,)

    Returns
        content_emb  : (1, T_frames, 1024)
        frame_lens   : (1,)
        acoustic_emb : (1, T_frames, 1024)
    """
    T         = wav.shape[1]
    attn_mask = (
        torch.arange(T, device=wav.device).unsqueeze(0) < audio_len.unsqueeze(1)
    ).long()
    out    = se_model.wavlm(wav, attention_mask=attn_mask, output_hidden_states=True)
    hidden = out.hidden_states

    content_emb  = hidden[se_model.content_layer]    # (1, T, 1024)
    acoustic_emb = hidden[se_model.acoustic_layer]   # (1, T, 1024)

    T_frames   = content_emb.size(1)
    frame_lens = se_model.frame_lengths(audio_len).clamp(max=T_frames)

    return content_emb, frame_lens, acoustic_emb


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_sample(se_model, decoder, generator, noisy_path: str,
               device: torch.device, fs: int = 16000):
    wav_np = librosa.load(noisy_path, sr=fs, mono=True)[0]   # (T,)
    wav    = torch.from_numpy(wav_np).unsqueeze(0).to(device) # (1, T)
    alen   = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)

    # ── 1. WavLM encode ───────────────────────────────────────────────────────
    content_emb, frame_lens, acoustic_emb = encode(se_model, wav, alen)

    # ── 2. Build decoder input (no CTC: proj_content applied directly) ────────
    acoustic_h = se_model.proj_acoustic(acoustic_emb)   # (1, T, 1024)
    content_h  = se_model.proj_content(content_emb)     # (1, T, 1024)
    input_feat = acoustic_h + content_h

    # ── 3. TransformerDecoder → mel ───────────────────────────────────────────
    transformer_out = decoder(input_feat, frame_lens)    # (1, T, 1024)
    mel_pred        = decoder.mel_head(transformer_out)  # (1, T, n_mels)

    # ── 4. Vocoder ────────────────────────────────────────────────────────────
    enh_wav = generator(mel_pred)  # (1, T_samples)

    return enh_wav[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference (no CTC): noisy → enhanced speech, saved by reverb/SNR"
    )
    p.add_argument("-C", "--config",
                   default=str(_HERE / "configs" / "config_stage2.yaml"))
    p.add_argument("--ckpt",    required=True,
                   help="Path to trained checkpoint .tar")
    p.add_argument("--se_cfg", default=None,
                   help="Path to se_model config yaml (fallback if ckpt has no se_cfg key)")
    p.add_argument("--filelist",
                   default="/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt")
    p.add_argument("--output_dir", default="enhanced_wo_ctc",
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

    fs = config["samplerate"]

    print(f"[ckpt]   {args.ckpt}")
    print(f"[output] {args.output_dir}")

    # ── load se_cfg (fallback; checkpoint's "se_cfg" key takes priority) ────────
    se_cfg = None
    if args.se_cfg:
        try:
            with open(args.se_cfg) as f:
                se_cfg = yaml.safe_load(f)
            print(f"[se_cfg] loaded from {args.se_cfg} (overridden by ckpt if present)")
        except FileNotFoundError:
            print(f"[se_cfg] {args.se_cfg} not found; will use ckpt's embedded se_cfg")

    # ── load models ───────────────────────────────────────────────────────────
    se_model, decoder, generator = build_models(args.ckpt, config, device, se_cfg=se_cfg)

    # ── read filelist ─────────────────────────────────────────────────────────
    samples = []
    with open(args.filelist) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            noisy_path = parts[2]
            snr        = int(float(parts[4])) if len(parts) > 4 else 0
            if "with_reverb" in noisy_path:
                reverb = "with_reverb"
            elif ("without_reverb" in noisy_path) or ("no_reverb" in noisy_path):
                reverb = "no_reverb"
            else:
                reverb = "unknown"
            clean_path = parts[0]
            samples.append((clean_path, noisy_path, reverb, snr))

    print(f"[filelist] {len(samples)} samples from {args.filelist}")

    # ── run inference ─────────────────────────────────────────────────────────
    for clean_path, noisy_path, reverb, snr in tqdm(samples, desc="inference", dynamic_ncols=True):
        enh = run_sample(
            se_model, decoder, generator,
            noisy_path=noisy_path,
            device=device,
            fs=fs,
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
