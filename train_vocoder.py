"""
Standalone vocoder training: clean mel → clean speech
  - WavLMDec receives log-mel (n_mels) directly (input_channels = n_mels)
  - GAN training with MPD + MBD discriminators
  - Dataset: clean audio only (uses clean channel from the noisy-clean filelist)

Checkpoint format:
  { epoch, generator, discriminator,
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
# Dataset  (clean audio only)
# ─────────────────────────────────────────────────────────────────────────────

class CleanDataset(data.Dataset):
    """
    Reads the same filelist format as NoisyCleanDataset
    (clean | noise | noisy | text) but loads only the clean channel.
    """

    def __init__(
        self,
        filelist: str,
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
                self.meta.append({
                    "id":    f"fileid_{i}",
                    "stem":  Path(parts[0]).stem,
                    "clean": parts[0],
                })
        print(f"Loaded {len(self.meta)} clean samples from {filelist}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        info = self.meta[idx]
        fs   = self.default_fs
        clean = librosa.load(info["clean"], sr=fs, mono=True)[0][np.newaxis, :]
        orig_len = min(clean.shape[1], self.max_audio_len)
        clean = clean[:, :orig_len]
        scale = 0.9 / (np.max(np.abs(clean)) + 1e-12)
        clean = (clean * scale).astype(np.float32)
        return clean, np.int64(orig_len), {"id": info["id"], "stem": info["stem"], "fs": fs}

    @staticmethod
    def collate_fn(batch):
        clean_list, lengths, infos = zip(*batch)
        max_len = max(c.shape[1] for c in clean_list)
        B = len(clean_list)
        clean_pad = np.zeros((B, 1, max_len), dtype=np.float32)
        for i, c in enumerate(clean_list):
            T = c.shape[1]
            clean_pad[i, :, :T] = c
        return (
            torch.from_numpy(clean_pad),
            torch.tensor(lengths, dtype=torch.long),
            list(infos),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class VocoderTrainer:
    def __init__(self, config, generator, discriminator,
                 optimizer_g, optimizer_d,
                 scheduler_g, scheduler_d,
                 loss_func, mel_loss_fn,
                 train_loader, val_loader, args):

        self.config        = config
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
        with open(os.path.join(self.exp_path, "config_vocoder.yaml"), "w") as f:
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

        self.writer         = SummaryWriter(self.log_path)
        self.start_epoch    = 1
        self.best_score     = 1e8
        self.best_mel_score = 1e8
        self.best_state     = None

        if tr["resume"]:
            self._resume_checkpoint()

    # ── mode helpers ──────────────────────────────────────────────────────────

    def _set_train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def _set_eval_mode(self):
        self.generator.eval()
        self.discriminator.eval()

    # ── checkpointing ─────────────────────────────────────────────────────────

    def _state(self):
        return {
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
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        print(f"[resume] Loaded {ckpts[-1]} (epoch {ckpt['epoch']})")

    # ── crop ─────────────────────────────────────────────────────────────────

    def _crop(self, latent, clean_wav, frame_lens, seg_frames, hop, random_crop):
        B = latent.size(0)
        out_list, clean_list = [], []
        for i in range(B):
            vf      = min(int(frame_lens[i].item()), latent.size(1))
            clean_i = clean_wav[i, :, :vf * hop]

            if vf >= seg_frames and random_crop:
                start_f = random.randint(0, vf - seg_frames)
            else:
                start_f = 0

            l = latent[i, start_f: start_f + seg_frames]
            c = clean_i[:, start_f * hop: (start_f + seg_frames) * hop]

            if l.size(0) < seg_frames:
                l = F.pad(l, (0, 0, 0, seg_frames - l.size(0)))
            if c.size(-1) < seg_frames * hop:
                c = F.pad(c, (0, seg_frames * hop - c.size(-1)))

            out_list.append(l)
            clean_list.append(c)

        return torch.stack(out_list), torch.stack(clean_list)

    # ── forward ───────────────────────────────────────────────────────────────

    def _forward(self, clean_wav, audio_lengths, random_crop=True):
        hop        = self.config["vocoder_config"]["hop_length"]
        seg_frames = int(self.config["wav_len"] * self.default_fs) // hop

        # Compute log-mel from clean waveform: (B, n_mels, T_frames) → (B, T_frames, n_mels)
        mel = self.mel_transform(clean_wav.squeeze(1))
        mel = (mel + 1e-5).log().transpose(1, 2)

        T_frames   = mel.size(1)
        frame_lens = (audio_lengths / hop).long().clamp(max=T_frames)

        # Crop to training segment length
        mel_c, clean_c = self._crop(
            mel, clean_wav, frame_lens, seg_frames, hop, random_crop
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

        for step, (clean_wav, audio_lengths, _) in enumerate(bar, 1):
            clean_wav     = clean_wav.to(self.device)
            audio_lengths = audio_lengths.to(self.device)

            esti_wav, clean_wav, valid_wav_lens = self._forward(
                clean_wav, audio_lengths, random_crop=True
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
                self.generator.parameters(),
                self.clip_grad_norm_value,
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

        for step, (clean_wav, audio_lengths, infos) in enumerate(bar, 1):
            clean_wav     = clean_wav.to(self.device)
            audio_lengths = audio_lengths.to(self.device)

            esti_wav, clean_wav, valid_wav_lens = self._forward(
                clean_wav, audio_lengths, random_crop=False
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
            self.writer.add_image(f"mel/esti_{i}",  self._mel_to_img(esti_mel),  epoch)
            self.writer.add_image(f"mel/clean_{i}", self._mel_to_img(clean_mel), epoch)
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
        print(f"Vocoder training for {self.epochs} epochs done.")


# ─────────────────────────────────────────────────────────────────────────────
# Build and run
# ─────────────────────────────────────────────────────────────────────────────

def build_models(config, device):
    voc_cfg = config["vocoder_config"]

    generator = WavLMDec(
        cond_dim  =None,
        cond_mode ="concat",
        spk_dim   =None,
        **{k: v for k, v in voc_cfg.items() if k not in ("cond_mode",)},
    ).to(device)

    # Optionally initialise from a pretrained vocoder checkpoint
    pretrained = config.get("pretrained_path", "")
    if pretrained:
        print(f"[pretrained] Loading vocoder from: {pretrained}")
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
        key  = "generator" if "generator" in ckpt else "model"
        generator.load_state_dict(ckpt[key], strict=False)

    disc_cfg = config["discriminator_config"]
    mpd = MultiPeriodDiscriminator(**disc_cfg["mpd"]).to(device)
    mbd = MultiBandDiscriminator(**disc_cfg["mbd"]).to(device)
    discriminator = CombinedDiscriminator([mpd, mbd]).to(device)

    return generator, discriminator


def run(config, args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.device = device

    generator, discriminator = build_models(config, device)

    train_ds = CleanDataset(**config["train_dataset"])
    val_ds   = CleanDataset(**config["validation_dataset"])

    train_loader = data.DataLoader(
        train_ds, collate_fn=CleanDataset.collate_fn,
        **config["train_dataloader"], shuffle=True,
    )
    val_loader = data.DataLoader(
        val_ds, collate_fn=CleanDataset.collate_fn,
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

    trainer = VocoderTrainer(
        config=config,
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
        description="Standalone vocoder training: clean mel → clean speech"
    )
    p.add_argument("-C", "--config",  default=str(_HERE / "configs" / "config_vocoder.yaml"))
    p.add_argument("-D", "--device",  default=0, type=int)
    p.add_argument("-R", "--resume",  action="store_true")
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
