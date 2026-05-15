"""
Stage-1 training: WavLMLoRASE + TransformerDecoder
  - Main loss  : mel_l1  (L1 on log-mel from decoder.mel_head vs clean reference)
  - Aux losses : CTC, speaker cosine similarity
  - No vocoder, no discriminator

Checkpoint format:
  { epoch, se_cfg, se_model, decoder, optimizer_se, optimizer_dec,
    scheduler_se, scheduler_dec }
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

class Stage1Trainer:
    def __init__(self, config, se_model, decoder,
                 optimizer_se, optimizer_dec,
                 scheduler_se, scheduler_dec,
                 train_loader, val_loader, args):

        self.config       = config
        self.se_model     = se_model
        self.decoder      = decoder
        self.optimizer_se  = optimizer_se
        self.optimizer_dec = optimizer_dec
        self.scheduler_se  = scheduler_se
        self.scheduler_dec = scheduler_dec
        self.train_loader = train_loader
        self.val_loader   = val_loader

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
        with open(os.path.join(self.exp_path, "config_stage1.yaml"), "w") as f:
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
        self.best_state  = None

        if tr["resume"]:
            self._resume_checkpoint()

    # ── mode helpers ──────────────────────────────────────────────────────────

    def _set_train_mode(self):
        self.se_model.train()
        self.decoder.train()

    def _set_eval_mode(self):
        self.se_model.eval()
        self.decoder.eval()

    # ── checkpointing ─────────────────────────────────────────────────────────

    def _state(self):
        return {
            "se_cfg":       self.se_model.cfg,
            "se_model":     self.se_model.state_dict(),
            "decoder":      self.decoder.state_dict(),
            "optimizer_se":  self.optimizer_se.state_dict(),
            "optimizer_dec": self.optimizer_dec.state_dict(),
            "scheduler_se":  self.scheduler_se.state_dict(),
            "scheduler_dec": self.scheduler_dec.state_dict(),
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
        self.se_model.load_state_dict(ckpt["se_model"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.optimizer_se.load_state_dict(ckpt["optimizer_se"])
        self.optimizer_dec.load_state_dict(ckpt["optimizer_dec"])
        self.scheduler_se.load_state_dict(ckpt["scheduler_se"])
        self.scheduler_dec.load_state_dict(ckpt["scheduler_dec"])
        print(f"[resume] Loaded {ckpts[-1]} (epoch {ckpt['epoch']})")

    # ── encode helper ─────────────────────────────────────────────────────────

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

    # ── forward ───────────────────────────────────────────────────────────────

    def _forward(self, noisy_wav, clean_wav, audio_lengths,
                 text_ids=None, text_lengths=None):
        hop = self.config["vocoder_config"]["hop_length"]

        content_emb, frame_lens, acoustic_emb = self._encode(noisy_wav, audio_lengths)

        content_fc = self.se_model.ctc_proj(content_emb)

        # CTC loss
        use_ctc = self.cond in ("content", "both")
        if use_ctc and text_ids is not None and text_lengths is not None:
            log_prob = F.log_softmax(content_fc, dim=-1).permute(1, 0, 2).clamp(min=-100.0)
            ctc_loss = self.se_model.ctc_loss_fn(
                log_prob, text_ids.to(log_prob.device),
                frame_lens, text_lengths.to(log_prob.device),
            )
        else:
            ctc_loss = torch.zeros((), device=noisy_wav.device)

        # Build decoder input
        acoustic_h = self.se_model.proj_acoustic(acoustic_emb)
        if self.cond in ("content", "both"):
            content_h  = self.se_model.proj_content(content_fc)
            input_feat = acoustic_h + content_h
        else:
            input_feat = acoustic_h

        transformer_out = self.decoder(input_feat, frame_lens)

        # Mel prediction and L1 loss
        T          = transformer_out.size(1)
        mel_pred   = self.decoder.mel_head(transformer_out)
        target_mel = self.mel_transform(
            clean_wav[:, :, :T * hop].squeeze(1)
        )
        target_mel = (target_mel + 1e-5).log().transpose(1, 2)
        T_min      = min(mel_pred.size(1), target_mel.size(1))
        frame_mask = (
            torch.arange(T_min, device=transformer_out.device).unsqueeze(0)
            < frame_lens.clamp(max=T_min).unsqueeze(1)
        )
        mel_l1_loss = F.l1_loss(
            mel_pred[:, :T_min][frame_mask],
            target_mel[:, :T_min].detach()[frame_mask],
        )

        return (
            mel_l1_loss, ctc_loss,
            mel_pred[:, :T_min].detach(),
            target_mel[:, :T_min].detach(),
            frame_lens,
        )

    # ── train epoch ───────────────────────────────────────────────────────────

    def _train_epoch(self, epoch):
        coeff = self.config["coeff"]
        lam_mel_l1 = coeff.get("mel_l1", 1.0)
        lam_ctc    = coeff.get("ctc",    0.0)

        totals = {"mel_l1": 0.0, "ctc": 0.0, "total": 0.0}
        bar = tqdm(self.train_loader, dynamic_ncols=True, mininterval=5.0,
                   desc=f"  train[{epoch}/{self.epochs + self.start_epoch - 1}]")

        for step, (noisy_wav, clean_wav, audio_lengths, text_ids, text_lengths, infos) in enumerate(bar, 1):
            noisy_wav     = noisy_wav.squeeze(1).to(self.device)
            clean_wav     = clean_wav.to(self.device)
            audio_lengths = audio_lengths.to(self.device)
            if text_ids is not None:
                text_ids     = text_ids.to(self.device)
                text_lengths = text_lengths.to(self.device)

            mel_l1_loss, ctc_loss, _, _, _ = self._forward(
                noisy_wav, clean_wav, audio_lengths,
                text_ids=text_ids, text_lengths=text_lengths,
            )

            loss = lam_mel_l1 * mel_l1_loss + lam_ctc * ctc_loss

            self.optimizer_se.zero_grad()
            self.optimizer_dec.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.se_model.parameters() if p.requires_grad]
                + list(self.decoder.parameters()),
                self.clip_grad_norm_value,
            )
            self.optimizer_se.step()
            self.optimizer_dec.step()
            self.scheduler_se.step()
            self.scheduler_dec.step()

            totals["mel_l1"] += mel_l1_loss.item()
            totals["ctc"]    += ctc_loss.item()
            totals["total"]  += loss.item()

        self.writer.add_scalars("lr", {
            "se":  self.optimizer_se.param_groups[0]["lr"],
            "dec": self.optimizer_dec.param_groups[0]["lr"],
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
        coeff = self.config["coeff"]
        lam_mel_l1 = coeff.get("mel_l1", 1.0)
        lam_ctc    = coeff.get("ctc",    0.0)

        totals = {"mel_l1": 0.0, "ctc": 0.0, "total": 0.0}
        bar = tqdm(self.val_loader, dynamic_ncols=True, mininterval=5.0,
                   desc=f"validate[{epoch}/{self.epochs + self.start_epoch - 1}]")

        N_VIS = 4
        vis_samples = []

        for step, (noisy_wav, clean_wav, audio_lengths, text_ids, text_lengths, infos) in enumerate(bar, 1):
            noisy_wav     = noisy_wav.squeeze(1).to(self.device)
            clean_wav     = clean_wav.to(self.device)
            audio_lengths = audio_lengths.to(self.device)
            if text_ids is not None:
                text_ids     = text_ids.to(self.device)
                text_lengths = text_lengths.to(self.device)

            mel_l1_loss, ctc_loss, mel_pred_vis, target_mel_vis, frame_lens_vis = (
                self._forward(noisy_wav, clean_wav, audio_lengths,
                              text_ids=text_ids, text_lengths=text_lengths)
            )

            loss = lam_mel_l1 * mel_l1_loss + lam_ctc * ctc_loss

            totals["mel_l1"] += mel_l1_loss.item()
            totals["ctc"]    += ctc_loss.item()
            totals["total"]  += loss.item()

            if len(vis_samples) < N_VIS:
                for i in range(mel_pred_vis.size(0)):
                    if len(vis_samples) >= N_VIS:
                        break
                    fl = int(frame_lens_vis[i].item())
                    vis_samples.append({
                        "mel_pred":   mel_pred_vis[i, :fl],
                        "mel_target": target_mel_vis[i, :fl],
                    })

        for i, s in enumerate(vis_samples):
            self.writer.add_image(f"mel_pred/{i}",   self._mel_to_img(s["mel_pred"]),   epoch)
            self.writer.add_image(f"mel_target/{i}", self._mel_to_img(s["mel_target"]), epoch)

        self.writer.add_scalars("val_loss", {k: v / step for k, v in totals.items()}, epoch)
        return totals["total"] / step, totals["mel_l1"] / step

    # ── main loop ─────────────────────────────────────────────────────────────

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            val_loss, val_mel_l1 = self._validation_epoch(epoch)
            torch.cuda.empty_cache()

            self._save_checkpoint(epoch, val_mel_l1)
            self._del_checkpoint(epoch)

            print(f"[epoch {epoch}] val_total={val_loss:.5f}  val_mel_l1={val_mel_l1:.5f}")

        if self.best_state is not None:
            torch.save(
                self.best_state,
                os.path.join(
                    self.ckpt_path,
                    f"best_model_{str(self.best_state['epoch']).zfill(3)}.tar",
                ),
            )
        print(f"Stage-1 training for {self.epochs} epochs done.")


# ─────────────────────────────────────────────────────────────────────────────
# Build and run
# ─────────────────────────────────────────────────────────────────────────────

def build_models(config, args, device):
    pretrained_path = config.get("pretrained_path", "")

    if pretrained_path:
        print(f"[pretrained] Loading checkpoint: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        se_cfg = ckpt.get("se_cfg") or ckpt.get("cfg")
        if se_cfg is None:
            if args.se_cfg is None:
                raise ValueError("Checkpoint has no 'se_cfg'/'cfg' key. Please provide --se_cfg.")
            with open(args.se_cfg) as f:
                se_cfg = yaml.safe_load(f)
    else:
        if args.se_cfg is None:
            raise ValueError("--se_cfg is required when pretrained_path is not set.")
        with open(args.se_cfg) as f:
            se_cfg = yaml.safe_load(f)
        ckpt = None

    se_model = WavLMLoRASE(se_cfg).to(device)
    if ckpt is not None:
        missing, unexpected = se_model.load_state_dict(ckpt["se_model"], strict=False)
        if missing:
            print(f"[pretrained] Missing keys: {missing}")
        if unexpected:
            print(f"[pretrained] Unexpected keys: {unexpected}")

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

    if ckpt is not None and "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
        print("[pretrained] Decoder loaded from checkpoint.")

    return se_model, decoder, se_cfg


def run(config, args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.device = device

    se_model, decoder, se_cfg = build_models(config, args, device)

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

    optimizer_se = AdamW(
        [p for p in se_model.parameters() if p.requires_grad],
        lr=opt_cfg["se"]["lr"], betas=betas, weight_decay=wd,
    )
    scheduler_se = WarmupLR(optimizer_se,
        warmup_steps=warmup_steps, decay_until_step=decay_until_step,
        max_lr=opt_cfg["se"]["max_lr"], min_lr=opt_cfg["se"]["min_lr"],
    )

    optimizer_dec = AdamW(
        decoder.parameters(),
        lr=opt_cfg["dec"]["lr"], betas=betas, weight_decay=wd,
    )
    scheduler_dec = WarmupLR(optimizer_dec,
        warmup_steps=warmup_steps, decay_until_step=decay_until_step,
        max_lr=opt_cfg["dec"]["max_lr"], min_lr=opt_cfg["dec"]["min_lr"],
    )

    trainer = Stage1Trainer(
        config=config,
        se_model=se_model, decoder=decoder,
        optimizer_se=optimizer_se, optimizer_dec=optimizer_dec,
        scheduler_se=scheduler_se, scheduler_dec=scheduler_dec,
        train_loader=train_loader, val_loader=val_loader,
        args=args,
    )
    trainer.train()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage-1: WavLMLoRASE + TransformerDecoder (mel_l1 main loss)")
    p.add_argument("-C", "--config",  default=str(_HERE / "configs" / "config_stage1.yaml"))
    p.add_argument("-D", "--device",  default=0, type=int)
    p.add_argument("-R", "--resume",  action="store_true")
    p.add_argument("--se_cfg",        default="config_latent.yaml",
                   help="Path to se_model config yaml (needed when pretrained_path has no cfg key)")
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
