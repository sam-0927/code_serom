"""
Dataset for speech enhancement training.

Filelist format (5-col, space-separated ' | '):
  clean_path | noise_path | noisy_path | text | snr

  col 0 : clean waveform  (LibriSpeech FLAC, used as GT)
  col 2 : noisy waveform  (model input)
  col 3 : transcript text (used for CTC loss)

Speaker ID is extracted from the noisy filename:
  {speaker_id}-{chapter}-{utterance}.wav → first '-'-separated field
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'serom'))

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def text_to_ids(text: str) -> list[int]:
    text = text.lower()
    return [CHAR2ID[c] for c in text if c in CHAR2ID]

# ── CTC vocabulary ────────────────────────────────────────────────────────────
# 0: blank | 1-26: a-z | 27: space | 28: apostrophe
CHAR2ID = {c: i + 1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
CHAR2ID[' '] = 27
CHAR2ID["'"] = 28
BLANK_ID = 0
ID2CHAR  = {v: k for k, v in CHAR2ID.items()}
ID2CHAR[BLANK_ID] = ''

# ─────────────────────────────────────────────────────────────────────────────
# Speaker helpers
# ─────────────────────────────────────────────────────────────────────────────

def speaker_from_path(path: str) -> str:
    """Extract speaker ID from noisy filename: first '-'-separated field."""
    return os.path.basename(path).split('-')[0]


def build_spk2int(*filelist_paths: str) -> dict[str, int]:
    """Scan one or more filelists (col[2] noisy path) and build speaker → int mapping."""
    speakers: set[str] = set()
    for filelist_path in filelist_paths:
        with open(filelist_path) as f:
            for line in f:
                parts = line.strip().split(' | ')
                if len(parts) < 3:
                    continue
                speakers.add(speaker_from_path(parts[2].strip()))
    return {s: i for i, s in enumerate(sorted(speakers))}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SEDataset(Dataset):
    """
    Returns (noisy_wav, clean_wav, text_ids, speaker_label, audio_length).

    noisy_wav / clean_wav : 1-D float32 tensors, truncated to max_audio_len
    text_ids              : 1-D int64 tensor of CTC target IDs
    speaker_label         : int64 scalar
    audio_length          : int64 scalar (actual noisy_wav length)
    """

    def __init__(
        self,
        filelist_path:   str,
        spk2int:         dict[str, int],
        spk_dir_path:    str,
        latent_dir_path: str,
        max_audio_len:   int = 240000,
        max_text_len:    int = 300,
        sample_rate:     int = 16000,
    ):
        self.spk2int       = spk2int
        self.max_audio_len = max_audio_len
        self.sample_rate   = sample_rate
        self.spk_dir_path  = spk_dir_path
        self.latent_dir_path = latent_dir_path
        # (noisy_path, clean_path, text_ids_tensor, spk_idx, clean_stem)
        self.samples: list[tuple[str, str, torch.Tensor, int, str]] = []

        skipped = 0
        with open(filelist_path) as f:
            for line in f:
                parts = line.strip().split(' | ')
                if len(parts) < 4:
                    continue
                clean_path = parts[0].strip()
                noisy_path = parts[2].strip()
                text       = parts[3].strip().lower()

                ids = text_to_ids(text)
                if not ids or len(ids) > max_text_len:
                    skipped += 1
                    continue

                spk_id = speaker_from_path(noisy_path)
                if spk_id not in spk2int:
                    skipped += 1
                    continue

                clean_stem = os.path.splitext(os.path.basename(clean_path))[0]
                self.samples.append((
                    noisy_path, clean_path,
                    torch.tensor(ids, dtype=torch.long),
                    spk2int[spk_id],
                    clean_stem,
                ))

        if skipped:
            print(f"[SEDataset] Skipped {skipped} samples")
        print(f"[SEDataset] {len(self.samples)} samples  ← {filelist_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        noisy_path, clean_path, text_ids, spk_idx, clean_stem = self.samples[idx]

        noisy_wav = self._load(noisy_path)
        clean_wav = self._load(clean_path)

        length    = min(noisy_wav.size(0), clean_wav.size(0), self.max_audio_len)
        noisy_wav = noisy_wav[:length]
        clean_wav = clean_wav[:length]

        spk_emb = torch.load(os.path.join(self.spk_dir_path, clean_stem+'.pt'), weights_only=True)
        clean_latent_emb = np.load(os.path.join(self.latent_dir_path, clean_stem+'_latent.npy'))

        return (
            noisy_wav,
            clean_wav,
            text_ids,
            torch.tensor(spk_idx, dtype=torch.long),
            torch.tensor(length,  dtype=torch.long),
            spk_emb,
            torch.tensor(clean_latent_emb),
        )

    def _load(self, path: str) -> torch.Tensor:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            g = gcd(sr, self.sample_rate)
            wav_np = resample_poly(wav_np, self.sample_rate // g, sr // g, axis=0)
        wav = torch.from_numpy(wav_np.T.astype(np.float32))   # (C, T)
        return wav.mean(0)   # mono, (T,)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_se(batch):
    """
    Returns
    -------
    noisy_pad        : (B, T_max_audio)
    clean_pad        : (B, T_max_audio)
    texts_pad        : (B, S_max)
    spk_labels       : (B,)
    audio_lengths    : (B,)
    text_lengths     : (B,)
    """
    noisy_wavs, clean_wavs, text_ids, spk_labels, audio_lengths, spk_embs, latent_embs = zip(*batch)

    noisy_pad     = pad_sequence(noisy_wavs,  batch_first=True)
    clean_pad     = pad_sequence(clean_wavs,  batch_first=True)
    texts_pad     = pad_sequence(text_ids,    batch_first=True, padding_value=BLANK_ID)
    spk_labels    = torch.stack(spk_labels)
    audio_lengths = torch.stack(audio_lengths)
    text_lengths  = torch.tensor([t.size(0) for t in text_ids], dtype=torch.long)
    spk_embs      = torch.stack(list(spk_embs))
    latent_pad    = pad_sequence(list(latent_embs), batch_first=True)

    return noisy_pad, clean_pad, texts_pad, spk_labels, spk_embs, latent_pad, audio_lengths, text_lengths


# ─────────────────────────────────────────────────────────────────────────────
# EER trial dataset  (for validation)
# ─────────────────────────────────────────────────────────────────────────────

class TrialDataset(Dataset):
    """
    Reads a trial file:  label path1 path2
    Embeds all unique utterances for EER scoring.
    """
    def __init__(self, trial_file: str, max_audio_len: int = 240000,
                 sample_rate: int = 16000):
        self.max_audio_len = max_audio_len
        self.sample_rate   = sample_rate

        self.trials: list[tuple[int, str, str]] = []
        paths_set: set[str] = set()
        with open(trial_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(maxsplit=2)
                label, p1, p2 = int(parts[0]), parts[1], parts[2]
                self.trials.append((label, p1, p2))
                paths_set.update([p1, p2])

        self.unique_paths = sorted(paths_set)
        self.path2idx     = {p: i for i, p in enumerate(self.unique_paths)}

    def __len__(self) -> int:
        return len(self.unique_paths)

    def __getitem__(self, idx: int):
        path = self.unique_paths[idx]
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != self.sample_rate:
            g = gcd(sr, self.sample_rate)
            wav_np = resample_poly(wav_np, self.sample_rate // g, sr // g, axis=0)
        wav = torch.from_numpy(wav_np.mean(axis=1).astype(np.float32))
        wav = wav[:self.max_audio_len]
        return wav, torch.tensor(wav.size(0), dtype=torch.long)


def collate_trial(batch):
    wavs, lengths = zip(*batch)
    return pad_sequence(wavs, batch_first=True), torch.stack(lengths)
