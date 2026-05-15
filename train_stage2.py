"""
Stage-2 training: Vocoder only (se_model + decoder frozen from stage-1)
  - Loads stage-1 checkpoint and freezes se_model + decoder
  - Trains generator (WavLMDec) + discriminator (MPD + MBD)
  - Losses: MultiScaleMelSpectrogramLoss (recons), adversarial, feature matching

Checkpoint format:
  { epoch, se_cfg, se_model, decoder,
    generator, discriminator,
    optimizer_g, optimizer_d, scheduler_g, scheduler_d }
"""

import os
import sys
import random
import shutil
import argparse
from copy import deepcopy
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torchaudio
from torch.optim import AdamW
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from wavlm_lora import WavLMLoRASE
from datasets import text_to_ids, BLANK_ID
from conformer_decoder import TransformerDecoder
from vocoder.wavlmdec import WavLMDec
from vocoder.discriminators import (
    MultiPeriodDiscriminator,
    MultiBandDiscriminator,
    CombinedDiscriminator,
)
from utils.loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)
from losses import MelSpectrogramLoss
from utils.scheduler import LinearWarmupCosineAnnealingLR as WarmupLR

# ── reproducibility ───────────────────────────────────────────────────────────
seed = 43
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NoisyCleanDataset(data.Dataset):
    def __init__(
        self,
        filelist: str,
        num_per_epoch: int = 10000,
        default_fs: int = 16000,
        max_audio_len: int = 240000,
    ):
        self.default_fs    = default_fs
        self.max_audio_len = max_audio_len

        self.meta = []
        with open(filelist) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                text = parts[3].strip().lower() if len(parts) > 3 else ""
                ids  = text_to_ids(text) if text else []
                self.meta.append({
                    "id":       f"fileid_{i}",
                    "stem":     Path(parts[0]).stem,
                    "clean":    parts[0],
                    "noisy":    parts[2],
                    "text_ids": torch.tensor(ids, dtype=torch.long) if ids else None,
                })
        print(f"Loaded {len(self.meta)} samples from {filelist}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        info = self.meta[idx]
        fs   = self.default_fs
        clean = librosa.load(info["clean"], sr=fs, mono=True)[0][np.newaxis, :]
        noisy = librosa.load(info["noisy"], sr=fs, mono=True)[0][np.newaxis, :]
        orig_len = min(clean.shape[1], noisy.shape[1], self.max_audio_len)
        clean, noisy = clean[:, :orig_len], noisy[:, :orig_len]
        scale = 0.9 / (max(np.max(np.abs(noisy)), np.max(np.abs(clean))) + 1e-12)
        return (
            (noisy * scale).astype(np.float32),
            (clean * scale).astype(np.float32),
            np.int64(orig_len),
            info["text_ids"],
            {"id": info["id"], "stem": info["stem"], "fs": fs},
        )

    @staticmethod
    def collate_fn(batch):
        noisy_list, clean_list, lengths, text_ids_list, infos = zip(*batch)
        max_len = max(n.shape[1] for n in noisy_list)
        B = len(noisy_list)
        noisy_pad = np.zeros((B, 1, max_len), dtype=np.float32)
        clean_pad = np.zeros((B, 1, max_len), dtype=np.float32)
        for i, (n, c) in enumerate(zip(noisy_list, clean_list)):
            T = n.shape[1]
            noisy_pad[i, :, :T] = n
            clean_pad[i, :, :T] = c

        valid = [t for t in text_ids_list if t is not None]
        if valid:
            text_lengths = torch.tensor(
                [t.size(0) if t is not None else 0 for t in text_ids_list],
                dtype=torch.long,
            )
            max_tlen  = max(t.size(0) for t in valid)
            texts_pad = torch.full((B, max_tlen), BLANK_ID, dtype=torch.long)
            for i, t in enumerate(text_ids_list):
                if t is not None:
                    texts_pad[i, :t.size(0)] = t
        else:
            texts_pad    = None
            text_lengths = torch.zeros(B, dtype=torch.long)

        return (
            torch.from_numpy(noisy_pad),
            torch.from_numpy(clean_pad),
            torch.tensor(lengths, dtype=torch.long),
            texts_pad,
            text_lengths,
            list(infos),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Trainer:
    def __init__(self, config, se_model, decoder, generator, discriminator,
                 optimizer_g, optimizer_d,
                 scheduler_g, scheduler_d,
                 loss_func, mel_loss_fn,
                 train_loader, val_loader, args):

        self.config        = config
        self.se_model      = se_model
        self.decoder       = decoder
        self.generator     = generator
        self.discriminator = discriminator
        self.optimizer_g   = optimizer_g
        self.optimizer_d   = optimizer_d
        self.scheduler_g   = scheduler_g
        self.scheduler_d   = scheduler_d
        self.loss_func     = loss_func
        self.mel_loss_fn   = mel_loss_fn
        self.train_loader  = train_loader
        self.val_loader    = val_loader

        self.device     = args.device
        self.cond       = config["cond"]
        self.default_fs = config["samplerate"]

        tr = config["trainer"]
        self.epochs                   = tr["epochs"]
        self.save_checkpoint_interval = tr["save_checkpoint_interval"]
        self.clip_grad_norm_value     = tr["clip_grad_norm_value"]

        self.exp_path    = tr["exp_path"]
        self.log_path    = os.path.join(self.exp_path, "logs")
        self.ckpt_path   = os.path.join(self.exp_path, "checkpoints")
        self.sample_path = os.path.join(self.exp_path, "val_samples")
        self.code_path   = os.path.join(self.exp_path, "codes")

        for p in [self.log_path, self.ckpt_path, self.sample_path, self.code_path]:
            os.makedirs(p, exist_ok=True)

        shutil.copy2(__file__, self.exp_path)
        with open(os.path.join(self.exp_path, "config_stage2.yaml"), "w") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        with open(os.path.join(self.exp_path, "cmd.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        for src_file in Path(__file__).parent.iterdir():
            if src_file.is_file():
                shutil.copy2(src_file, self.code_path)
        for d in ["configs", "vocoder", "utils"]:
            src = Path(__file__).parent / d
            if src.exists():
                shutil.copytree(src, Path(self.code_path) / d, dirs_exist_ok=True)

        voc_cfg = config["vocoder_config"]
        n_mels  = config.get("n_mels", 80)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["samplerate"],
            n_fft      =voc_cfg["n_fft"],
            hop_length =voc_cfg["hop_length"],
            win_length =voc_cfg["n_fft"],
            n_mels     =n_mels,
        ).to(self.device)

        self.writer      = SummaryWriter(self.log_path)
        self.start_epoch = 1
        self.best_score  = 1e8
        self.best_mel_score = 1e8
        self.best_state  = None

        if tr["resume"]:
            self._resume_checkpoint()

    # ── mode helpers ──────────────────────────────────────────────────────────

    def _set_train_mode(self):
        # se_model and decoder stay in eval mode (frozen)
        self.se_model.eval()
        self.decoder.eval()
        self.generator.train()
        self.discriminator.train()

    def _set_eval_mode(self):
        self.se_model.eval()
        self.decoder.eval()
        self.generator.eval()
        self.discriminator.eval()

    # ── checkpointing ─────────────────────────────────────────────────────────

    def _state(self):
        return {
            "se_cfg":        self.se_model.cfg,
            "se_model":      self.se_model.state_dict(),
            "decoder":       self.decoder.state_dict(),
            "generator":     self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g":   self.optimizer_g.state_dict(),
            "optimizer_d":   self.optimizer_d.state_dict(),
            "scheduler_g":   self.scheduler_g.state_dict(),
            "scheduler_d":   self.scheduler_d.state_dict(),
        }

    def _save_checkpoint(self, epoch, score):
        state = {"epoch": epoch, **self._state()}
        torch.save(state, os.path.join(self.ckpt_path, f"model_{str(epoch).zfill(3)}.tar"))
        if score < self.best_score:
            self.best_state = deepcopy(state)
            self.best_score = score

    def _del_checkpoint(self, epoch):
        if (epoch - 1) % self.save_checkpoint_interval != 0:
            prev = os.path.join(self.ckpt_path, f"model_{str(epoch-1).zfill(3)}.tar")
            if os.path.exists(prev):
                os.remove(prev)

    def _resume_checkpoint(self):
        ckpts = sorted(glob(os.path.join(self.ckpt_path, "model_*.tar")))
        if not ckpts:
            print("[resume] No checkpoint found, starting from scratch.")
            return
        ckpt = torch.load(ckpts[-1], map_location=self.device, weights_only=False)
        self.start_epoch = ckpt["epoch"] + 1
        # se_model / decoder states are included for completeness; they remain frozen
        self.se_model.load_state_dict(ckpt["se_model"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"[resume] Loaded {ckpts[-1]} (epoch {ckpt['epoch']})")

    # ── frozen forward helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _encode(self, wav: torch.Tensor, audio_lengths: torch.Tensor):
        T         = wav.shape[1]
        attn_mask = (
            torch.arange(T, device=wav.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
        ).long()
        out    = self.se_model.wavlm(wav, attention_mask=attn_mask, output_hidden_states=True)
        hidden = out.hidden_states

        content_emb  = hidden[self.se_model.content_layer]
        acoustic_emb = hidden[self.se_model.acoustic_layer]

        T_frames   = content_emb.size(1)
        frame_lens = self.se_model.frame_lengths(audio_lengths).clamp(max=T_frames)

        return content_emb, frame_lens, acoustic_emb

    @torch.no_grad()
    def _frozen_forward(self, noisy_wav, audio_lengths):
        """Run frozen se_model + decoder; returns predicted mel (80-dim) and frame lengths."""
        content_emb, frame_lens, acoustic_emb = self._encode(noisy_wav, audio_lengths)

        content_fc = self.se_model.ctc_proj(content_emb)
        acoustic_h = self.se_model.proj_acoustic(acoustic_emb)
        if self.cond in ("content", "both"):
            content_h  = self.se_model.proj_content(content_fc)
            input_feat = acoustic_h + content_h
        else:
            input_feat = acoustic_h

        transformer_out = self.decoder(input_feat, frame_lens)
        mel_pred        = self.decoder.mel_head(transformer_out)   # (B, T, n_mels)
        return mel_pred, frame_lens

    # ── crop ─────────────────────────────────────────────────────────────────

    def _crop(self, mel, clean_wav, frame_lens, seg_frames, hop, random_crop):
        B = mel.size(0)
        out_list, clean_list = [], []
        for i in range(B):
            vf      = min(int(frame_lens[i].item()), mel.size(1))
            clean_i = clean_wav[i, :, :vf * hop]

            if vf >= seg_frames and random_crop:
                start_f = random.randint(0, vf - seg_frames)
            else:
                start_f = 0

            m = mel[i, start_f: start_f + seg_frames]
            c = clean_i[:, start_f * hop: (start_f + seg_frames) * hop]

            if m.size(0) < seg_frames:
                m = F.pad(m, (0, 0, 0, seg_frames - m.size(0)))
            if c.size(-1) < seg_frames * hop:
                c = F.pad(c, (0, seg_frames * hop - c.size(-1)))

            out_list.append(m)
            clean_list.append(c)

        return torch.stack(out_list), torch.stack(clean_list)

    # ── forward ───────────────────────────────────────────────────────────────

    def _forward(self, noisy_wav, clean_wav, audio_lengths, random_crop=True):
        hop        = self.config["vocoder_config"]["hop_length"]
        seg_frames = int(self.config["wav_len"] * self.default_fs) // hop

        mel_pred, frame_lens = self._frozen_forward(noisy_wav, audio_lengths)

        T         = mel_pred.size(1)
        clean_wav = clean_wav[:, :, :T * hop]

        mel_c, clean_c = self._crop(
            mel_pred, clean_wav, frame_lens, seg_frames, hop, random_crop
        )

        valid_wav_lens = frame_lens.clamp(max=seg_frames) * hop
        esti_wav       = self.generator(mel_c)

        return esti_wav, clean_c, valid_wav_lens

    # ── train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, epoch):
        coeff      = self.config["coeff"]
        lam_recons = coeff["recons"]
        lam_adv    = coeff["adv"]
        lam_feat   = coeff["feat"]

        totals = {"loss": 0.0, "adv": 0.0, "feat": 0.0, "dis": 0.0, "mel": 0.0}
        bar = tqdm(self.train_loader, dynamic_ncols=True, mininterval=5.0,
                   desc=f"  train[{epoch}/{self.epochs + self.start_epoch - 1}]")

        for step, (noisy_wav, clean_wav, audio_lengths, _, _, _) in enumerate(bar, 1):
            noisy_wav     = noisy_wav.squeeze(1).to(self.device)
            clean_wav     = clean_wav.to(self.device)
            audio_lengths = audio_lengths.to(self.device)

            esti_wav, clean_wav, valid_wav_lens = self._forward(
                noisy_wav, clean_wav, audio_lengths, random_crop=True
            )

            if esti_wav.ndim == 1:
                esti_wav = esti_wav.unsqueeze(0)
            esti_wav = esti_wav.unsqueeze(1)

            L    = esti_wav.shape[-1]
            mask = (
                torch.arange(L, device=self.device).unsqueeze(0)
                < valid_wav_lens.unsqueeze(1)
            ).unsqueeze(1).float()
            esti_w  = esti_wav  * mask
            clean_w = clean_wav * mask

            # ── generator step ────────────────────────────────────────────────
            loss_mel  = lam_recons * self.loss_func(esti_w, clean_w)
            _, esti_metric, true_fmap, esti_fmap = self.discriminator(clean_w, esti_w)
            loss_adv  = lam_adv * generator_loss(esti_metric)[0]
            loss_feat = lam_feat * feature_loss(true_fmap, esti_fmap)
            loss_g    = loss_mel + loss_adv + loss_feat

            self.optimizer_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.clip_grad_norm_value
            )
            self.optimizer_g.step()

            # ── discriminator step ────────────────────────────────────────────
            true_metric_d, esti_metric_d, _, _ = self.discriminator(clean_w, esti_w.detach())
            loss_dis = discriminator_loss(true_metric_d, esti_metric_d)[0]

            self.optimizer_d.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.clip_grad_norm_value
            )
            self.optimizer_d.step()

            self.scheduler_g.step()
            self.scheduler_d.step()

            with torch.no_grad():
                loss_mel_simple = self.mel_loss_fn(esti_w.squeeze(1), clean_w.squeeze(1))

            totals["loss"] += loss_g.item()
            totals["adv"]  += loss_adv.item()
            totals["feat"] += loss_feat.item()
            totals["dis"]  += loss_dis.item()
            totals["mel"]  += loss_mel_simple.item()

        self.writer.add_scalars("lr", {
            "g": self.optimizer_g.param_groups[0]["lr"],
            "d": self.optimizer_d.param_groups[0]["lr"],
        }, epoch)
        self.writer.add_scalars("train_loss", {k: v / step for k, v in totals.items()}, epoch)

    # ── mel image helper ──────────────────────────────────────────────────────

    @staticmethod
    def _mel_to_img(mel: torch.Tensor) -> torch.Tensor:
        m = mel.float().cpu().T.flip(0)
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        return m.unsqueeze(0)

    # ── validation epoch ──────────────────────────────────────────────────────

    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        coeff      = self.config["coeff"]
        lam_recons = coeff["recons"]
        lam_adv    = coeff["adv"]
        lam_feat   = coeff["feat"]

        totals = {"loss": 0.0, "adv": 0.0, "feat": 0.0, "dis": 0.0, "mel": 0.0}
        bar = tqdm(self.val_loader, dynamic_ncols=True, mininterval=5.0,
                   desc=f"validate[{epoch}/{self.epochs + self.start_epoch - 1}]")

        N_VIS = 4
        vis_samples = []

        for step, (noisy_wav, clean_wav, audio_lengths, _, _, infos) in enumerate(bar, 1):
            noisy_wav     = noisy_wav.squeeze(1).to(self.device)
            clean_wav     = clean_wav.to(self.device)
            audio_lengths = audio_lengths.to(self.device)

            esti_wav, clean_wav, valid_wav_lens = self._forward(
                noisy_wav, clean_wav, audio_lengths, random_crop=False
            )

            if esti_wav.ndim == 1:
                esti_wav = esti_wav.unsqueeze(0)
            esti_wav = esti_wav.unsqueeze(1)

            L    = esti_wav.shape[-1]
            mask = (
                torch.arange(L, device=self.device).unsqueeze(0)
                < valid_wav_lens.unsqueeze(1)
            ).unsqueeze(1).float()
            esti_w  = esti_wav  * mask
            clean_w = clean_wav * mask

            loss_mel  = lam_recons * self.loss_func(esti_w, clean_w)
            _, esti_metric, true_fmap, esti_fmap = self.discriminator(clean_w, esti_w)
            loss_adv  = lam_adv * generator_loss(esti_metric)[0]
            loss_feat = lam_feat * feature_loss(true_fmap, esti_fmap)
            loss_g    = loss_mel + loss_adv + loss_feat

            true_metric_d, esti_metric_d, _, _ = self.discriminator(clean_w, esti_w)
            loss_dis = discriminator_loss(true_metric_d, esti_metric_d)[0]

            loss_mel_simple = self.mel_loss_fn(esti_w.squeeze(1), clean_w.squeeze(1))

            totals["loss"] += loss_g.item()
            totals["adv"]  += loss_adv.item()
            totals["feat"] += loss_feat.item()
            totals["dis"]  += loss_dis.item()
            totals["mel"]  += loss_mel_simple.item()

            if len(vis_samples) < N_VIS:
                for i in range(esti_w.size(0)):
                    if len(vis_samples) >= N_VIS:
                        break
                    vl = int(valid_wav_lens[i].item())
                    vis_samples.append({
                        "esti_w":  esti_w[i, 0, :vl].cpu(),
                        "clean_w": clean_w[i, 0, :vl].cpu(),
                        "vl":      vl,
                    })

            if (epoch < 10 or epoch % 10 == 0) and step <= 5:
                uid = infos[0]["id"]
                sf.write(
                    os.path.join(self.sample_path, f"{uid}_clean.wav"),
                    clean_w[0, 0].cpu().numpy(), self.default_fs,
                )
                sf.write(
                    os.path.join(self.sample_path, f"{uid}_esti_ep{str(epoch).zfill(3)}.wav"),
                    esti_w[0, 0].cpu().numpy(), self.default_fs,
                )

        for i, s in enumerate(vis_samples):
            esti_mel  = (self.mel_transform(s["esti_w"].unsqueeze(0).to(self.device))
                         + 1e-5).log().squeeze(0).T
            clean_mel = (self.mel_transform(s["clean_w"].unsqueeze(0).to(self.device))
                         + 1e-5).log().squeeze(0).T
            self.writer.add_image(f"vocos_mel/esti_{i}",  self._mel_to_img(esti_mel),  epoch)
            self.writer.add_image(f"vocos_mel/clean_{i}", self._mel_to_img(clean_mel), epoch)
            self.writer.add_audio(f"audio/esti_{i}",  s["esti_w"],  epoch, sample_rate=self.default_fs)
            self.writer.add_audio(f"audio/clean_{i}", s["clean_w"], epoch, sample_rate=self.default_fs)

        self.writer.add_scalars("val_loss", {k: v / step for k, v in totals.items()}, epoch)
        return totals["loss"] / step, totals["mel"] / step

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            val_loss, val_mel = self._validation_epoch(epoch)
            torch.cuda.empty_cache()

            self._save_checkpoint(epoch, val_loss)
            self._del_checkpoint(epoch)

            if val_mel < self.best_mel_score:
                self.best_mel_score = val_mel
                state = {"epoch": epoch, **self._state()}
                torch.save(state, os.path.join(self.ckpt_path, "best_mel_model.tar"))
                print(f"[best_mel] epoch {epoch}  mel={val_mel:.6f} → saved")

        if self.best_state is not None:
            torch.save(
                self.best_state,
                os.path.join(
                    self.ckpt_path,
                    f"best_model_{str(self.best_state['epoch']).zfill(3)}.tar",
                ),
            )
        print(f"Stage-2 training for {self.epochs} epochs done.")


# ─────────────────────────────────────────────────────────────────────────────
# Build and run
# ─────────────────────────────────────────────────────────────────────────────

def build_models(config, args, device):
    # ── Load stage-1 checkpoint to restore se_model + decoder ────────────────
    stage1_ckpt_path = config.get("stage1_ckpt_path", "")
    if not stage1_ckpt_path:
        raise ValueError("config['stage1_ckpt_path'] must point to a stage-1 checkpoint.")

    print(f"[stage1] Loading frozen models from: {stage1_ckpt_path}")
    stage1_ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)

    se_cfg = stage1_ckpt.get("se_cfg") or stage1_ckpt.get("cfg")
    if se_cfg is None:
        if args.se_cfg is None:
            raise ValueError("Stage-1 checkpoint has no 'se_cfg'. Provide --se_cfg.")
        with open(args.se_cfg) as f:
            se_cfg = yaml.safe_load(f)

    se_model = WavLMLoRASE(se_cfg).to(device)
    se_model.load_state_dict(stage1_ckpt["se_model"], strict=False)
    for p in se_model.parameters():
        p.requires_grad = False
    se_model.eval()
    print("[stage1] se_model frozen.")

    dec_cfg = se_cfg["decoder"]
    n_mels  = config.get("n_mels", 80)
    decoder = TransformerDecoder(
        input_dim=dec_cfg["d_model"],
        d_model  =dec_cfg["d_model"],
        n_layers =dec_cfg["n_layers"],
        n_heads  =dec_cfg["n_heads"],
        ffn_dim  =dec_cfg["ffn_dim"],
        dropout  =dec_cfg["dropout"],
        max_len  =dec_cfg.get("max_len", 4096),
        n_mels   =n_mels,
    ).to(device)
    decoder.load_state_dict(stage1_ckpt["decoder"])
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()
    print("[stage1] decoder frozen.")

    # ── Build trainable vocoder + discriminator ───────────────────────────────
    voc_cfg = {k: v for k, v in config["vocoder_config"].items()
               if k not in ("cond_mode",)}
    generator = WavLMDec(
        cond_dim  =None,
        cond_mode ="concat",
        spk_dim   =None,
        **voc_cfg,
    ).to(device)

    # Optionally load pretrained vocoder weights
    voc_pretrained = config.get("vocoder_pretrained_path", "")
    if voc_pretrained:
        print(f"[vocoder] Loading pretrained weights from: {voc_pretrained}")
        voc_ckpt = torch.load(voc_pretrained, map_location="cpu", weights_only=False)
        key = "generator" if "generator" in voc_ckpt else "model"
        generator.load_state_dict(voc_ckpt[key], strict=False)

    disc_cfg = config["discriminator_config"]
    mpd = MultiPeriodDiscriminator(**disc_cfg["mpd"]).to(device)
    mbd = MultiBandDiscriminator(**disc_cfg["mbd"]).to(device)
    discriminator = CombinedDiscriminator([mpd, mbd]).to(device)

    return se_model, decoder, generator, discriminator, se_cfg


def run(config, args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.device = device

    se_model, decoder, generator, discriminator, se_cfg = build_models(
        config, args, device
    )

    train_ds = NoisyCleanDataset(**config["train_dataset"])
    val_ds   = NoisyCleanDataset(**config["validation_dataset"])

    train_loader = data.DataLoader(
        train_ds, collate_fn=NoisyCleanDataset.collate_fn,
        **config["train_dataloader"], shuffle=True,
    )
    val_loader = data.DataLoader(
        val_ds, collate_fn=NoisyCleanDataset.collate_fn,
        **config["validation_dataloader"], shuffle=False,
    )

    opt_cfg          = config["optimizer"]
    betas            = tuple(opt_cfg["betas"])
    wd               = opt_cfg["weight_decay"]
    steps_per_epoch  = len(train_loader)
    warmup_steps     = steps_per_epoch * opt_cfg["warmup_epochs"]
    decay_until_step = steps_per_epoch * opt_cfg["decay_epochs"]
    print(f"[scheduler] steps_per_epoch={steps_per_epoch}  "
          f"warmup={warmup_steps}  decay_until={decay_until_step}")

    optimizer_g = AdamW(
        generator.parameters(),
        lr=opt_cfg["g"]["lr"], betas=betas, weight_decay=wd,
    )
    scheduler_g = WarmupLR(optimizer_g,
        warmup_steps=warmup_steps, decay_until_step=decay_until_step,
        max_lr=opt_cfg["g"]["max_lr"], min_lr=opt_cfg["g"]["min_lr"],
    )

    optimizer_d = AdamW(
        discriminator.parameters(),
        lr=opt_cfg["d"]["lr"], betas=betas, weight_decay=wd,
    )
    scheduler_d = WarmupLR(optimizer_d,
        warmup_steps=warmup_steps, decay_until_step=decay_until_step,
        max_lr=opt_cfg["d"]["max_lr"], min_lr=opt_cfg["d"]["min_lr"],
    )

    loss_func   = MultiScaleMelSpectrogramLoss(sampling_rate=config["samplerate"]).to(device)
    mel_loss_fn = MelSpectrogramLoss(sample_rate=config["samplerate"]).to(device)

    trainer = Stage2Trainer(
        config=config,
        se_model=se_model, decoder=decoder,
        generator=generator, discriminator=discriminator,
        optimizer_g=optimizer_g, optimizer_d=optimizer_d,
        scheduler_g=scheduler_g, scheduler_d=scheduler_d,
        loss_func=loss_func, mel_loss_fn=mel_loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        args=args,
    )
    trainer.train()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage-2: vocoder fine-tuning with frozen stage-1 front-end"
    )
    p.add_argument("-C", "--config",  default=str(_HERE / "configs" / "config_stage2.yaml"))
    p.add_argument("-D", "--device",  default=0, type=int)
    p.add_argument("-R", "--resume",  action="store_true")
    p.add_argument("--se_cfg",        default=None,
                   help="Path to se_model config yaml (fallback if stage1 ckpt has no cfg key)")
    p.add_argument("--task_name",     default=None)
    p.add_argument("--train_filelist",default=None)
    p.add_argument("--val_filelist",  default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.resume:
        config["trainer"]["resume"] = True
    if args.task_name is not None:
        config["trainer"]["exp_path"] = "outputs/" + args.task_name
    if args.train_filelist is not None:
        config["train_dataset"]["filelist"] = args.train_filelist
    if args.val_filelist is not None:
        config["validation_dataset"]["filelist"] = args.val_filelist

    run(config, args)
