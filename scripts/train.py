#!/usr/bin/env python
"""Shared training entry point for the proposed method and all baselines.

`--model proposed` trains the physics-conditioned diffusion model (the paper's
method). `--model {redcnn,hformer,ascon}` train supervised image-domain
denoisers taking c = A*(y) as input (the same conditioning quantity the
proposed method uses, for an apples-to-apples comparison rather than
whatever each baseline's own released code happens to assume). `--model
dugan` additionally trains the dual discriminators. `--model corediff` trains
the bridge-style generalized-diffusion baseline. `--model dps_prior` trains
the unconditional diffusion prior used by DPS (Sec. "Head-to-head comparison
with inference-time physics correction"): same UNet width/depth as `proposed`
but with no c conditioning and no physics loss; the physics correction is
applied only at evaluation time (see scripts/evaluate.py --model dps).
FBP needs no training and is not listed here -- it is a closed-form baseline
handled entirely in scripts/evaluate.py. See src/models/baselines/ for
fidelity caveats on each reimplementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ct.operators import CTProjector, ParallelBeamGeometry
from src.data.lodopab import LoDoPaBDataset
from src.data.mayo_aapm import MayoAAPMDataset
from src.data.transforms import AugmentConfig, random_augment
from src.diffusion.ddpm import PhysicsConditionedDDPM
from src.diffusion.dps import UnconditionalDDPM
from src.diffusion.schedule import make_ddpm_schedule
from src.losses.perceptual import VGGPerceptualLoss
from src.losses.physics import PhysicsConsistencyLoss
from src.models.baselines.ascon import ASCONDenoiser, ascon_contrastive_loss
from src.models.baselines.corediff import CoreDiffModel, make_generalized_schedule
from src.models.baselines.dugan import DUGANDiscriminators, DUGANGenerator
from src.models.baselines.hformer import Hformer
from src.models.baselines.redcnn import REDCNN
from src.models.unet import UNetConfig, UNetModel
from src.utils.io import ensure_dir, git_commit_hash, save_yaml
from src.utils.logging import make_tb_writer, setup_logger
from src.utils.seed import seed_everything, worker_init_fn

BASELINE_MODELS = ("redcnn", "hformer", "ascon", "dugan", "corediff", "dps_prior")
ALL_MODELS = ("proposed",) + BASELINE_MODELS


def build_dataset(cfg):
    if cfg.data.dataset == "lodopab":
        return LoDoPaBDataset(cfg.data.root, cfg.data.split)
    if cfg.data.dataset == "mayo_aapm":
        return MayoAAPMDataset(cfg.data.root, cfg.data.split)
    raise ValueError(f"Unknown dataset: {cfg.data.dataset}")


def build_supervised_model(model_name: str, image_size: int):
    if model_name == "redcnn":
        return REDCNN()
    if model_name == "hformer":
        return Hformer()
    raise ValueError(model_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--model", type=str, default="proposed", choices=ALL_MODELS)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    seed_everything(int(cfg.experiment.seed), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.experiment.out_dir) / f"{cfg.experiment.name}_{args.model}"
    ensure_dir(out_dir)
    log_dir = ensure_dir(out_dir / "logs")
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    logger = setup_logger(log_dir)
    writer = make_tb_writer(log_dir)

    cfg_meta = OmegaConf.to_container(cfg, resolve=True)
    cfg_meta["git_commit"] = git_commit_hash()
    cfg_meta["model"] = args.model
    save_yaml(cfg_meta, out_dir / "config_resolved.yaml")

    ds_train = build_dataset(OmegaConf.merge(cfg, {"data": {"split": "train"}}))
    ds_val = build_dataset(OmegaConf.merge(cfg, {"data": {"split": "val"}}))

    train_cfg = cfg.train
    dl_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    dl_val = DataLoader(
        ds_val, batch_size=1, shuffle=False, num_workers=int(cfg.data.num_workers),
        pin_memory=True, worker_init_fn=worker_init_fn,
    )

    H, W = cfg.data.image_size
    geom = ParallelBeamGeometry(image_size=int(H), angles=int(cfg.ct.angles), det_count=int(cfg.ct.det_count))
    projector = CTProjector(geom, device=device)

    grad_accum = int(cfg.optim.grad_accum_steps)
    max_epochs = int(cfg.optim.max_epochs)
    aug_cfg = AugmentConfig()

    if args.model == "proposed":
        _train_proposed(cfg, args, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs)
    elif args.model == "dps_prior":
        _train_dps_prior(cfg, device, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, int(H))
    elif args.model == "dugan":
        _train_dugan(cfg, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, int(H))
    elif args.model == "ascon":
        _train_ascon(cfg, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs)
    elif args.model == "corediff":
        _train_corediff(cfg, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, int(H))
    else:  # redcnn, hformer -- plain supervised image-domain denoisers
        _train_supervised(cfg, args.model, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, int(H))

    logger.info("Training finished.")


def _flush_grad_accum(scaler, opt, step: int, grad_accum: int, force: bool = False) -> None:
    """Fix for the old code's gradient-accumulation tail-drop: if the number
    of steps in an epoch isn't a multiple of grad_accum, the leftover
    accumulated gradient was silently discarded by the next epoch's
    zero_grad(). Call with force=True at the end of every epoch."""
    if force or (step + 1) % grad_accum == 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)


def _train_proposed(cfg, args, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs):
    H, W = cfg.data.image_size
    schedule = make_ddpm_schedule(
        T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule),
        beta_start=float(cfg.diffusion.beta_start), beta_end=float(cfg.diffusion.beta_end), device=device,
    )
    unet_cfg = UNetConfig(
        in_channels=int(cfg.model.in_channels), out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels), channel_mult=tuple(cfg.model.channel_mult),
        num_res_blocks=int(cfg.model.num_res_blocks), attention_resolutions=tuple(cfg.model.attention_resolutions),
        num_heads=int(cfg.model.num_heads), dropout=float(cfg.model.dropout),
    )
    denoiser = UNetModel(unet_cfg, image_size=int(H)).to(device)
    ddpm = PhysicsConditionedDDPM(denoiser, schedule).to(device)

    phys_loss_fn = PhysicsConsistencyLoss()
    perceptual = None
    if bool(cfg.loss.perceptual.enabled):
        perceptual = VGGPerceptualLoss(list(cfg.loss.perceptual.layers)).to(device)

    opt = torch.optim.AdamW(ddpm.parameters(), lr=float(cfg.optim.lr), betas=tuple(cfg.optim.betas), weight_decay=float(cfg.optim.weight_decay))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=str(cfg.optim.scheduler.mode), factor=float(cfg.optim.scheduler.factor), patience=int(cfg.optim.scheduler.patience), min_lr=float(cfg.optim.scheduler.min_lr))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))

    T = int(cfg.diffusion.T)
    physics_w = float(cfg.loss.physics_weight)
    perc_w = float(cfg.loss.perceptual.weight)
    # 0.0 realizes the "w/o denoising loss" ablation (Table: loss-term
    # ablation) as an actual code path rather than a no-op config override --
    # see the note in scripts/run_ablation.py this replaces.
    denoising_w = float(cfg.loss.get("denoising_weight", 1.0))
    best_val = float("inf")
    epochs_no_improve = 0

    def run_val(epoch: int) -> float:
        ddpm.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"val {epoch}", leave=False):
                x0 = batch["x"].to(device)
                y = batch["y"].to(device)
                x_min = batch["x_min"].to(device)
                x_max = batch["x_max"].to(device)
                if y.ndim == 2:
                    y = y.unsqueeze(0)
                c = projector.AT(y)
                t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
                eps = torch.randn_like(x0)
                x_t = ddpm.q_sample(x0, t, eps)
                out = ddpm.predict_eps_and_x0(x_t, c, t)
                loss = denoising_w * F.mse_loss(out.eps_pred, eps) + physics_w * phys_loss_fn(projector, out.x0_pred, y, x_min, x_max)
                if perceptual is not None:
                    loss = loss + perc_w * perceptual(out.x0_pred, x0)
                losses.append(loss.item())
        ddpm.train()
        return float(sum(losses) / max(1, len(losses)))

    for epoch in range(max_epochs):
        ddpm.train()
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl_train, desc=f"train {epoch}")
        step = 0
        for step, batch in enumerate(pbar):
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            x_min = batch["x_min"].to(device)
            x_max = batch["x_max"].to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)
            x0_aug = random_augment(x0, aug_cfg)
            c = projector.AT(y)
            t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
            eps = torch.randn_like(x0_aug)
            x_t = ddpm.q_sample(x0_aug, t, eps)

            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                out = ddpm.predict_eps_and_x0(x_t, c, t)
                loss_diff = denoising_w * F.mse_loss(out.eps_pred, eps)
                loss_phys = phys_loss_fn(projector, out.x0_pred, y, x_min, x_max)
                loss = loss_diff + physics_w * loss_phys
                if perceptual is not None:
                    loss = loss + perc_w * perceptual(out.x0_pred, x0_aug)

            scaler.scale(loss / grad_accum).backward()
            _flush_grad_accum(scaler, opt, step, grad_accum)
            pbar.set_postfix({"L": f"{loss.item():.4f}"})

        _flush_grad_accum(scaler, opt, step, grad_accum, force=True)

        if (epoch + 1) % int(cfg.train.val_every_epochs) == 0:
            val_loss = run_val(epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")
            sched.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({"model": ddpm.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg_meta}, ckpt_dir / "best.pt")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= int(cfg.optim.early_stop_patience):
                logger.info(f"Early stopping at epoch {epoch}.")
                break


def _train_dps_prior(cfg, device, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, image_size):
    """The DPS prior (Sec. "Head-to-head comparison with inference-time
    physics correction"): an UNconditional DDPM, same width/depth/attention
    as the proposed method's UNet (matched capacity) but with in_channels=1
    (no c = A*(y) concatenation) and trained with the plain denoising loss
    only -- no physics-consistency loss, no measurement conditioning. Physics
    is reintroduced only at evaluation time via scripts/evaluate.py's DPS
    sampler (src/diffusion/dps.py)."""
    schedule = make_ddpm_schedule(
        T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule),
        beta_start=float(cfg.diffusion.beta_start), beta_end=float(cfg.diffusion.beta_end), device=device,
    )
    unet_cfg = UNetConfig(
        in_channels=1, out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels), channel_mult=tuple(cfg.model.channel_mult),
        num_res_blocks=int(cfg.model.num_res_blocks), attention_resolutions=tuple(cfg.model.attention_resolutions),
        num_heads=int(cfg.model.num_heads), dropout=float(cfg.model.dropout),
    )
    denoiser = UNetModel(unet_cfg, image_size=image_size).to(device)
    model = UnconditionalDDPM(denoiser, schedule).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), betas=tuple(cfg.optim.betas), weight_decay=float(cfg.optim.weight_decay))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=str(cfg.optim.scheduler.mode), factor=float(cfg.optim.scheduler.factor), patience=int(cfg.optim.scheduler.patience), min_lr=float(cfg.optim.scheduler.min_lr))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))

    T = int(cfg.diffusion.T)
    best_val = float("inf")
    epochs_no_improve = 0

    def run_val(epoch: int) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"val[dps_prior] {epoch}", leave=False):
                x0 = batch["x"].to(device)
                t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
                eps = torch.randn_like(x0)
                x_t = model.q_sample(x0, t, eps)
                out = model.predict_eps_and_x0(x_t, t)
                losses.append(F.mse_loss(out.eps_pred, eps).item())
        model.train()
        return float(sum(losses) / max(1, len(losses)))

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(dl_train, desc=f"train[dps_prior] {epoch}")
        step = 0
        for step, batch in enumerate(pbar):
            x0 = batch["x"].to(device)
            x0_aug = random_augment(x0, aug_cfg)
            t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
            eps = torch.randn_like(x0_aug)
            x_t = model.q_sample(x0_aug, t, eps)

            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                out = model.predict_eps_and_x0(x_t, t)
                loss = F.mse_loss(out.eps_pred, eps)

            scaler.scale(loss / grad_accum).backward()
            _flush_grad_accum(scaler, opt, step, grad_accum)
            pbar.set_postfix({"L": f"{loss.item():.4f}"})

        _flush_grad_accum(scaler, opt, step, grad_accum, force=True)

        if (epoch + 1) % int(cfg.train.val_every_epochs) == 0:
            val_loss = run_val(epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            logger.info(f"[dps_prior] epoch {epoch}: val_loss={val_loss:.6f}")
            sched.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg_meta}, ckpt_dir / "best.pt")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= int(cfg.optim.early_stop_patience):
                logger.info(f"[dps_prior] Early stopping at epoch {epoch}.")
                break


def _train_supervised(cfg, model_name, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, image_size):
    """RED-CNN / Hformer: supervised image-domain denoising, input c = A*(y),
    target x0, plain pixel loss (+ optional perceptual)."""
    model = build_supervised_model(model_name, image_size).to(device)
    perceptual = None
    if bool(cfg.loss.perceptual.enabled):
        perceptual = VGGPerceptualLoss(list(cfg.loss.perceptual.layers)).to(device)
    perc_w = float(cfg.loss.perceptual.weight)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))
    best_val = float("inf")

    def run_val() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in dl_val:
                x0 = batch["x"].to(device)
                y = batch["y"].to(device)
                if y.ndim == 2:
                    y = y.unsqueeze(0)
                c = projector.AT(y)
                pred = model(c)
                losses.append(F.mse_loss(pred, x0).item())
        model.train()
        return float(sum(losses) / max(1, len(losses)))

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        step = 0
        for step, batch in enumerate(tqdm(dl_train, desc=f"train[{model_name}] {epoch}")):
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)
            x0_aug = random_augment(x0, aug_cfg)
            c = projector.AT(y)

            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                pred = model(c)
                loss = F.mse_loss(pred, x0_aug)
                if perceptual is not None:
                    loss = loss + perc_w * perceptual(pred, x0_aug)

            scaler.scale(loss / grad_accum).backward()
            _flush_grad_accum(scaler, opt, step, grad_accum)

        _flush_grad_accum(scaler, opt, step, grad_accum, force=True)

        val_loss = run_val()
        writer.add_scalar("val/loss", val_loss, epoch)
        logger.info(f"[{model_name}] epoch {epoch}: val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg_meta}, ckpt_dir / "best.pt")


def _train_dugan(cfg, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, image_size):
    gen = DUGANGenerator().to(device)
    disc = DUGANDiscriminators().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=float(cfg.optim.lr), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(cfg.optim.lr), betas=(0.5, 0.999))
    best_val = float("inf")

    def run_val() -> float:
        gen.eval()
        losses = []
        with torch.no_grad():
            for batch in dl_val:
                x0 = batch["x"].to(device)
                y = batch["y"].to(device)
                if y.ndim == 2:
                    y = y.unsqueeze(0)
                c = projector.AT(y)
                pred = gen(c)
                losses.append(F.mse_loss(pred, x0).item())
        gen.train()
        return float(sum(losses) / max(1, len(losses)))

    for epoch in range(max_epochs):
        gen.train()
        disc.train()
        for batch in tqdm(dl_train, desc=f"train[dugan] {epoch}"):
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)
            x0_aug = random_augment(x0, aug_cfg)
            c = projector.AT(y)

            fake = gen(c)

            # Discriminator step (LSGAN-style least-squares adversarial loss).
            opt_d.zero_grad(set_to_none=True)
            real_img, real_grad = disc(x0_aug)
            fake_img, fake_grad = disc(fake.detach())
            d_loss = (
                F.mse_loss(real_img, torch.ones_like(real_img)) + F.mse_loss(fake_img, torch.zeros_like(fake_img))
                + F.mse_loss(real_grad, torch.ones_like(real_grad)) + F.mse_loss(fake_grad, torch.zeros_like(fake_grad))
            ) * 0.5
            d_loss.backward()
            opt_d.step()

            # Generator step: adversarial + pixel reconstruction loss.
            opt_g.zero_grad(set_to_none=True)
            fake_img, fake_grad = disc(fake)
            g_adv = F.mse_loss(fake_img, torch.ones_like(fake_img)) + F.mse_loss(fake_grad, torch.ones_like(fake_grad))
            g_pix = F.mse_loss(fake, x0_aug)
            g_loss = g_pix + 0.1 * g_adv
            g_loss.backward()
            opt_g.step()

        val_loss = run_val()
        writer.add_scalar("val/loss", val_loss, epoch)
        logger.info(f"[dugan] epoch {epoch}: val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"generator": gen.state_dict(), "discriminators": disc.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg_meta}, ckpt_dir / "best.pt")


def _train_ascon(cfg, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs):
    model = ASCONDenoiser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))
    contrastive_w = 0.1
    best_val = float("inf")

    def run_val() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in dl_val:
                x0 = batch["x"].to(device)
                y = batch["y"].to(device)
                if y.ndim == 2:
                    y = y.unsqueeze(0)
                c = projector.AT(y)
                pred = model(c)
                losses.append(F.mse_loss(pred, x0).item())
        model.train()
        return float(sum(losses) / max(1, len(losses)))

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        step = 0
        for step, batch in enumerate(tqdm(dl_train, desc=f"train[ascon] {epoch}")):
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)
            x0_aug = random_augment(x0, aug_cfg)
            c = projector.AT(y)

            # Two-view contrastive setup (see ascon.py fidelity caveat):
            # each item and its independently-augmented twin form a positive
            # pair; patch_ids = original batch index repeated twice.
            c2 = projector.AT(y)  # same measurement, re-used as the "second view" input
            x0_aug2 = random_augment(x0, aug_cfg)

            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                pred1, emb1 = model(c, return_embedding=True)
                pred2, emb2 = model(c2, return_embedding=True)
                pix_loss = F.mse_loss(pred1, x0_aug) + F.mse_loss(pred2, x0_aug2)

                b = c.shape[0]
                patch_ids = torch.arange(b, device=device).repeat(2)
                embeddings = torch.cat([emb1, emb2], dim=0)
                contrastive_loss = ascon_contrastive_loss(embeddings, patch_ids)
                loss = pix_loss + contrastive_w * contrastive_loss

            scaler.scale(loss / grad_accum).backward()
            _flush_grad_accum(scaler, opt, step, grad_accum)

        _flush_grad_accum(scaler, opt, step, grad_accum, force=True)

        val_loss = run_val()
        writer.add_scalar("val/loss", val_loss, epoch)
        logger.info(f"[ascon] epoch {epoch}: val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg_meta}, ckpt_dir / "best.pt")


def _train_corediff(cfg, device, projector, dl_train, dl_val, ckpt_dir, logger, writer, cfg_meta, aug_cfg, grad_accum, max_epochs, image_size):
    T = int(cfg.diffusion.T)
    schedule = make_generalized_schedule(T, device=device)
    model = CoreDiffModel(image_size=image_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))
    best_val = float("inf")

    def make_xt(x0, x_ldct, t_idx):
        gamma = schedule.gammas[t_idx].view(-1, 1, 1, 1)
        sigma = schedule.sigmas[t_idx].view(-1, 1, 1, 1)
        return (1 - gamma) * x0 + gamma * x_ldct + sigma * torch.randn_like(x0)

    def run_val() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in dl_val:
                x0 = batch["x"].to(device)
                y = batch["y"].to(device)
                if y.ndim == 2:
                    y = y.unsqueeze(0)
                c = projector.AT(y)
                t_idx = torch.randint(0, T, (x0.shape[0],), device=device)
                x_t = make_xt(x0, c, t_idx)
                x0_pred = model.predict_x0(x_t, c, t_idx)
                losses.append(F.mse_loss(x0_pred, x0).item())
        model.train()
        return float(sum(losses) / max(1, len(losses)))

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        step = 0
        for step, batch in enumerate(tqdm(dl_train, desc=f"train[corediff] {epoch}")):
            x0 = batch["x"].to(device)
            y = batch["y"].to(device)
            if y.ndim == 2:
                y = y.unsqueeze(0)
            x0_aug = random_augment(x0, aug_cfg)
            c = projector.AT(y)
            t_idx = torch.randint(0, T, (x0.shape[0],), device=device)

            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                x_t = make_xt(x0_aug, c, t_idx)
                x0_pred = model.predict_x0(x_t, c, t_idx)
                loss = F.mse_loss(x0_pred, x0_aug)

            scaler.scale(loss / grad_accum).backward()
            _flush_grad_accum(scaler, opt, step, grad_accum)

        _flush_grad_accum(scaler, opt, step, grad_accum, force=True)

        val_loss = run_val()
        writer.add_scalar("val/loss", val_loss, epoch)
        logger.info(f"[corediff] epoch {epoch}: val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg_meta}, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
