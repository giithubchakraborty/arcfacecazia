"""
Evaluate a trained checkpoint on an LFW-style verification set.

Usage:
    python eval.py --checkpoint checkpoints/epoch_31.pth
    python eval.py --checkpoint checkpoints/epoch_31.pth \
        --val_dir data/cfp_fp_112x112 --val_pairs data/cfp_fp_pairs.txt
"""
import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import VerificationPairs
from models import iresnet50, iresnet100
from utils import compute_embeddings, evaluate_verification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--val_pairs", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=["iresnet50", "iresnet100"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    val_dir = args.val_dir or cfg.val_dir
    val_pairs = args.val_pairs or cfg.val_pairs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    backbone_name = args.backbone or ckpt.get("backbone_name", cfg.backbone)
    embedding_size = ckpt.get("embedding_size", cfg.embedding_size)
    build_fn = iresnet50 if backbone_name == "iresnet50" else iresnet100
    backbone = build_fn(embedding_size=embedding_size).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()

    dataset = VerificationPairs(val_dir, val_pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    emb1, emb2, labels = compute_embeddings(backbone, loader, device)
    acc_mean, acc_std, thresh = evaluate_verification(emb1, emb2, labels)

    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Val set:     {val_dir}")
    print(f"Pairs:       {len(dataset)}")
    print(f"Accuracy:    {acc_mean * 100:.2f}% +/- {acc_std * 100:.2f}%")
    print(f"Mean thresh: {thresh:.3f} (cosine distance)")


if __name__ == "__main__":
    main()
