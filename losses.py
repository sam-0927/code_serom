"""
SE-specific loss functions

  si_snr_loss            : negative SI-SNR (scale-invariant SNR)
  MultiResolutionSTFTLoss: spectral magnitude L1 at multiple STFT resolutions
  waveform_l1_loss       : plain L1 on waveform samples (thin wrapper)
  MelSpectrogramLoss     : L1 on log-mel-spectrogram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ─────────────────────────────────────────────────────────────────────────────
# SI-SNR loss
# ─────────────────────────────────────────────────────────────────────────────

def si_snr_loss(estimated: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio loss.

    Parameters
    ----------
    estimated : (B, T)  generated waveform
    target    : (B, T)  clean reference waveform

    Returns
    -------
    Scalar loss = mean(-SI-SNR)  over batch.
    SI-SNR is negated so that minimising the loss maximises SI-SNR.
    """
    # Zero-mean
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target    = target    - target.mean(dim=-1, keepdim=True)

    # s_target: projection of estimated onto target
    dot        = (estimated * target).sum(dim=-1, keepdim=True)       # (B, 1)
    target_pow = (target * target).sum(dim=-1, keepdim=True) + eps    # (B, 1)
    s_target   = dot / target_pow * target                             # (B, T)

    # e_noise: residual
    e_noise = estimated - s_target                                     # (B, T)

    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )   # (B,)
    return -si_snr.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Single-resolution STFT magnitude loss
# ─────────────────────────────────────────────────────────────────────────────

def _stft_magnitude(x: torch.Tensor, n_fft: int, hop: int, win: int,
                    window: torch.Tensor) -> torch.Tensor:
    """Returns linear STFT magnitude (B, F, T)."""
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True,
    )
    return spec.abs()   # (B, n_fft//2+1, T_frames)


class STFTLoss(nn.Module):
    """
    Spectral convergence + log-magnitude L1 at a single STFT resolution.

    sc_loss  = ‖|STFT(y)| − |STFT(ŷ)|‖_F / ‖|STFT(y)|‖_F
    mag_loss = ‖ log|STFT(y)| − log|STFT(ŷ)| ‖_1 / numel
    """
    def __init__(self, n_fft: int, hop: int, win: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop   = hop
        self.win   = win
        self.register_buffer("window", torch.hann_window(win))

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mag_e = _stft_magnitude(estimated, self.n_fft, self.hop, self.win, self.window)
        mag_t = _stft_magnitude(target,    self.n_fft, self.hop, self.win, self.window)

        sc_loss  = (mag_t - mag_e).norm(p="fro") / (mag_t.norm(p="fro") + 1e-8)
        mag_loss = F.l1_loss(mag_e.clamp(min=1e-7).log(), mag_t.clamp(min=1e-7).log())
        return sc_loss + mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Average of STFTLoss over multiple (n_fft, hop, win) configurations.

    Default resolutions are chosen to cover short-to-long time scales
    at 16 kHz, following common SE / vocoder practice.
    """
    def __init__(self, resolutions: list[tuple[int, int, int]] | None = None):
        super().__init__()
        if resolutions is None:
            # (n_fft, hop_length, win_length)
            resolutions = [
                (256,  64,  256),
                (512,  128, 512),
                (1024, 256, 1024),
            ]
        self.losses = nn.ModuleList(
            [STFTLoss(n, h, w) for n, h, w in resolutions]
        )

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum(l(estimated, target) for l in self.losses) / len(self.losses)


# ─────────────────────────────────────────────────────────────────────────────
# Waveform L1
# ─────────────────────────────────────────────────────────────────────────────

def waveform_l1_loss(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise L1 on waveform samples."""
    return F.l1_loss(estimated, target)


# ─────────────────────────────────────────────────────────────────────────────
# Mel-spectrogram L1 loss
# ─────────────────────────────────────────────────────────────────────────────

class MelSpectrogramLoss(nn.Module):
    """
    L1 loss on log-mel-spectrogram.

    log_mel = log(MelSpectrogram(x) + eps)
    loss    = L1(log_mel_estimated, log_mel_target)

    Parameters
    ----------
    sample_rate  : audio sample rate (default 16000)
    n_fft        : FFT size
    hop_length   : hop size in samples
    win_length   : window size in samples
    n_mels       : number of mel filter banks
    f_min        : lowest mel frequency
    f_max        : highest mel frequency (None → sample_rate / 2)
    eps          : floor before log to avoid log(0)
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft:       int = 1024,
        hop_length:  int = 256,
        win_length:  int = 1024,
        n_mels:      int = 80,
        f_min:       float = 0.0,
        f_max:       float | None = None,
        eps:         float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
            power=1.0,          # amplitude spectrogram
        )

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        estimated : (B, T)  generated waveform
        target    : (B, T)  clean reference waveform

        Returns
        -------
        Scalar L1 loss on log-mel-spectrograms.
        """
        mel_e = self.mel_transform(estimated)           # (B, n_mels, T_frames)
        mel_t = self.mel_transform(target)
        log_mel_e = (mel_e + self.eps).log()
        log_mel_t = (mel_t + self.eps).log()
        return F.l1_loss(log_mel_e, log_mel_t)
