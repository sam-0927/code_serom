"""
Inference script for Stage-2 (no CTC) trained model
(WavLM + proj_content (no ctc_proj) + TransformerDecoder + WavLMDec vocoder).

Reads DNS-Challenge-2020 synthetic test set directly:
    <dns_root>/no_reverb/noisy/*.wav
    <dns_root>/no_reverb/clean/*.wav
    <dns_root>/with_reverb/noisy/*.wav
    <dns_root>/with_reverb/clean/*.wav

Noisy filename example : clnsp102_traffic_248091_3_snr0_tl-21_fileid_268.wav
Clean filename example  : clean_fileid_268.wav

Saves enhanced audio organized by reverb condition:

    <output_dir>/
        no_reverb/
            stem_enh.wav  stem_clean.wav ...
        with_reverb/
            ...
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

_DNS_ROOT = Path(
    "/workspace/DB/DNS-Challenge-2020/datasets/test_set/synthetic"
)


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_models(ckpt_path: str, config: dict, device: torch.device, se_cfg: dict = None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "se_cfg" in ckpt:
        se_cfg = ckpt["se_cfg"]
    elif se_cfg is None:
        raise ValueError("Checkpoint has no 'se_cfg' key. Please provide --se_cfg.")

    se_model = WavLMLoRASE(se_cfg).to(device)
    result = se_model.load_state_dict(ckpt["se_model"], strict=False)
    if result.missing_keys:
        print(f"[se_model] missing keys (randomly initialized): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[se_model] unexpected keys (ignored): {result.unexpected_keys}")
    se_model.eval()

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
    T         = wav.shape[1]
    attn_mask = (
        torch.arange(T, device=wav.device).unsqueeze(0) < audio_len.unsqueeze(1)
    ).long()
    out    = se_model.wavlm(wav, attention_mask=attn_mask, output_hidden_states=True)
    hidden = out.hidden_states

    content_emb  = hidden[se_model.content_layer]
    acoustic_emb = hidden[se_model.acoustic_layer]

    T_frames   = content_emb.size(1)
    frame_lens = se_model.frame_lengths(audio_len).clamp(max=T_frames)

    return content_emb, frame_lens, acoustic_emb


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_sample(se_model, decoder, generator, noisy_path: str,
               device: torch.device, fs: int = 16000):
    wav_np = librosa.load(noisy_path, sr=fs, mono=True)[0]
    wav    = torch.from_numpy(wav_np).unsqueeze(0).to(device)
    alen   = torch.tensor([wav.shape[1]], dtype=torch.long, device=device)

    content_emb, frame_lens, acoustic_emb = encode(se_model, wav, alen)

    # No CTC: proj_content applied directly to content_emb
    acoustic_h = se_model.proj_acoustic(acoustic_emb)
    content_h  = se_model.proj_content(content_emb)
    input_feat = acoustic_h + content_h

    transformer_out = decoder(input_feat, frame_lens)
    mel_pred        = decoder.mel_head(transformer_out)  # (1, T, n_mels)
    enh_wav         = generator(mel_pred)

    return enh_wav[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# DNS test-set scanner
# ─────────────────────────────────────────────────────────────────────────────

_FILEID_RE = re.compile(r"_fileid_(\d+)")


def collect_samples(dns_root: Path):
    samples = []
    for reverb_cond in ("no_reverb", "with_reverb"):
        noisy_dir = dns_root / reverb_cond / "noisy"
        clean_dir = dns_root / reverb_cond / "clean"

        if not noisy_dir.is_dir():
            print(f"[warn] directory not found: {noisy_dir}")
            continue

        clean_map = {}
        for p in clean_dir.glob("*.wav"):
            m = _FILEID_RE.search(p.stem)
            if m:
                clean_map[m.group(1)] = p

        for noisy_path in sorted(noisy_dir.glob("*.wav")):
            m_id = _FILEID_RE.search(noisy_path.stem)
            if not m_id:
                print(f"[warn] cannot parse fileid from {noisy_path.name}, skipping")
                continue
            fileid = m_id.group(1)

            clean_path = clean_map.get(fileid)
            if clean_path is None:
                print(f"[warn] no clean file for fileid={fileid}, skipping")
                continue

            samples.append((str(noisy_path), str(clean_path), reverb_cond))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference (no CTC) on DNS-Challenge-2020 synthetic test set"
    )
    p.add_argument("-C", "--config",
                   default=str(_HERE / "configs" / "config_stage2.yaml"))
    p.add_argument("--ckpt",    required=True,
                   help="Path to trained checkpoint .tar")
    p.add_argument("--se_cfg", default=None,
                   help="Fallback se_cfg yaml (used only if checkpoint lacks 'se_cfg' key)")
    p.add_argument("--dns_root", default=str(_DNS_ROOT),
                   help="Root of DNS synthetic test set "
                        "(contains no_reverb/ and with_reverb/)")
    p.add_argument("--output_dir", default="enhanced_dns_wo_ctc",
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

    print(f"[config] fs={fs}")
    print(f"[ckpt]   {args.ckpt}")
    print(f"[output] {args.output_dir}")

    se_cfg = None
    if args.se_cfg:
        try:
            with open(args.se_cfg) as f:
                se_cfg = yaml.safe_load(f)
            print(f"[se_cfg] loaded from {args.se_cfg} (overridden by ckpt if present)")
        except FileNotFoundError:
            print(f"[se_cfg] {args.se_cfg} not found; will use ckpt's embedded se_cfg")

    se_model, decoder, generator = build_models(args.ckpt, config, device, se_cfg=se_cfg)

    samples = collect_samples(Path(args.dns_root))
    print(f"[dataset] {len(samples)} samples from {args.dns_root}")

    for noisy_path, clean_path, reverb_cond in tqdm(
            samples, desc="inference", dynamic_ncols=True):

        enh = run_sample(
            se_model, decoder, generator,
            noisy_path=noisy_path,
            device=device,
            fs=fs,
        )

        out_dir = Path(args.output_dir) / reverb_cond
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(noisy_path).stem
        sf.write(str(out_dir / f"{stem}_enh.wav"),   enh,                                           fs)
        sf.write(str(out_dir / f"{stem}_clean.wav"), librosa.load(clean_path, sr=fs, mono=True)[0], fs)

    print("Done.")


if __name__ == "__main__":
    main()
