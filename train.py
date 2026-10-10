"""
Train ArcFace (Additive Angular Margin Loss) on CASIA-WebFace.

Usage:
    python train.py
    python train.py --batch_size 128 --epochs 32 --backbone iresnet50
    python train.py --resume checkpoints/epoch_10.pth

This reproduces the [CASIA, ResNet50, ArcFace] setting from Section 4.1 /
Table 2 of the paper. If val_dir + val_pairs (e.g. LFW) are available,
verification accuracy is reported every `val_interval` epochs.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config
from datasets import CASIAWebFace, VerificationPairs
from models import ArcMarginProduct, iresnet50, iresnet100
from utils import AverageMeter, compute_embeddings, evaluate_verification, save_checkpoint, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train ArcFace on CASIA-WebFace")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--val_pairs", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["iresnet50", "iresnet100"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="overrides the linear-scaled default")
    parser.add_argument("--s", type=float, default=None, help="ArcFace feature scale")
    parser.add_argument("--m", type=float, default=None, help="ArcFace angular margin")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg, args):
    for field in ("train_dir", "val_dir", "val_pairs", "backbone", "batch_size", "epochs", "num_workers", "resume"):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)
    if args.s is not None:
        cfg.s = args.s
    if args.m is not None:
        cfg.m = args.m
    if args.no_amp:
        cfg.use_amp = False
    cfg.explicit_lr = args.lr  # None unless the user passed --lr; see main()
    return cfg


def build_backbone(name, embedding_size, dropout):
    if name == "iresnet50":
        return iresnet50(embedding_size=embedding_size, dropout=dropout)
    if name == "iresnet100":
        return iresnet100(embedding_size=embedding_size, dropout=dropout)
    raise ValueError(f"Unknown backbone '{name}'")


def train_one_epoch(backbone, head, loader, criterion, optimizer, scaler, device, cfg, epoch, logger):
    backbone.train()
    head.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    start = time.time()
    use_amp = cfg.use_amp and device.type == "cuda"

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            embeddings = backbone(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if cfg.grad_clip_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), cfg.grad_clip_norm
            )
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            batch_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(batch_acc, images.size(0))

        if step % cfg.log_interval == 0:
            elapsed = time.time() - start
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"loss {loss_meter.avg:.4f} batch-acc {acc_meter.avg:.4f} "
                f"({elapsed:.1f}s elapsed)"
            )

    return loss_meter.avg, acc_meter.avg


def maybe_validate(backbone, cfg, device, logger, epoch):
    if not (cfg.val_dir and cfg.val_pairs and os.path.isdir(cfg.val_dir) and os.path.isfile(cfg.val_pairs)):
        return
    dataset = VerificationPairs(cfg.val_dir, cfg.val_pairs)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=cfg.num_workers)
    emb1, emb2, labels = compute_embeddings(backbone, loader, device)
    acc_mean, acc_std, thresh = evaluate_verification(emb1, emb2, labels)
    logger.info(f"[val @ epoch {epoch}] accuracy {acc_mean * 100:.2f}% +/- {acc_std * 100:.2f}% (thr {thresh:.3f})")
    backbone.train()


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_set = CASIAWebFace(cfg.train_dir, input_size=112)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    logger.info(f"Train set: {len(train_set)} images across {train_set.num_classes} identities")

    backbone = build_backbone(cfg.backbone, cfg.embedding_size, cfg.dropout).to(device)
    head = ArcMarginProduct(cfg.embedding_size, train_set.num_classes, s=cfg.s, m=cfg.m, easy_margin=cfg.easy_margin).to(device)
    criterion = nn.CrossEntropyLoss()

    if cfg.explicit_lr is not None:
        lr = cfg.explicit_lr
        lr_note = "explicit --lr, no scaling applied"
    else:
        lr = cfg.scaled_lr()
        lr_note = f"linearly scaled from base_lr={cfg.base_lr} @ bs=512"
    optimizer = optim.SGD(
        [{"params": backbone.parameters()}, {"params": head.parameters()}],
        lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    logger.info(f"Optimizer: SGD lr={lr:.4f} ({lr_note}) "
                f"momentum={cfg.momentum} weight_decay={cfg.weight_decay}")
    logger.info(f"ArcFace head: s={cfg.s} m={cfg.m}")

    start_epoch = 0
    if cfg.resume and os.path.isfile(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from {cfg.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        loss, acc = train_one_epoch(backbone, head, train_loader, criterion, optimizer, scaler, device, cfg, epoch, logger)
        scheduler.step()
        logger.info(f"Epoch [{epoch}] done. avg loss {loss:.4f}, avg batch-acc {acc:.4f}, lr now {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % cfg.val_interval == 0:
            maybe_validate(backbone, cfg, device, logger, epoch)

        if (epoch + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}.pth")
            save_checkpoint({
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "backbone_name": cfg.backbone,
                "embedding_size": cfg.embedding_size,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
