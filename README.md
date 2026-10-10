# ArcFace on CASIA-WebFace

A from-scratch PyTorch implementation of the base **ArcFace** method from

> J. Deng, J. Guo, J. Yang, N. Xue, I. Kotsia, S. Zafeiriou,
> "ArcFace: Additive Angular Margin Loss for Deep Face Recognition,"
> arXiv:1801.07698.

trained on **CASIA-WebFace at 112x112**, matching the `[CASIA, ResNet50,
ArcFace]` setup the paper itself uses for its Table 2 ablation.

## What's implemented

- **IResNet50 / IResNet100 backbone** (`models/iresnet.py`) — the "ResNet50 /
  ResNet100 without the bottleneck structure" plus the "BN-Dropout-FC-BN"
  embedding head described in Section 4.1.
- **ArcFace loss** (`models/metrics.py`) — the additive angular margin head,
  Eq. 2-3 of the paper: normalize embeddings and class-center weights, add
  margin `m` to the target angle, rescale by `s`, softmax cross-entropy.
- **CASIA-WebFace data loading** (`datasets/casia_webface.py`) — a
  folder-per-identity `Dataset` with the light augmentation (random
  horizontal flip) and normalization standard for ArcFace training.
- **Verification evaluation** (`datasets/verification.py`,
  `utils/lfw_eval.py`) — the standard 10-fold LFW protocol used for the
  paper's LFW / CFP-FP / AgeDB / CALFW / CPLFW numbers (Table 2, Table 5).
- **Training loop** (`train.py`) with the paper's CASIA schedule (SGD,
  lr=0.1, divided by 10 at epochs 20 and 28, 32 epochs total, s=64, m=0.5),
  checkpointing, resuming, and optional mixed precision.

### Not implemented (out of scope for this request)

The paper also proposes **sub-center ArcFace** (Section 3.2) and a
**model-inversion / face-generation** method (Section 3.3). Both are
separate contributions built on top of base ArcFace:

- Sub-center ArcFace targets *noisy, large-scale* web data (MS1MV0,
  Celeb500K) — the paper itself trains it on those, not on the clean,
  small-scale CASIA-WebFace. Adding it would mean adding `K` sub-centers per
  class in `ArcMarginProduct`, a max-pool over sub-centers, and an offline
  step that drops non-dominant sub-centers/high-confidence noisy samples
  (Eq. 7, Section 3.2) — architecturally a straightforward extension of
  `models/metrics.py` if you later train on noisier data.
- The inversion/generation experiments (Section 3.3, Eq. 8) reconstruct face
  images from a *trained* ArcFace model's gradients and BN statistics; it's
  a separate downstream use of the checkpoints this repo produces, not part
  of training the recognition model itself.

## Project layout

```
arcface_casia/
├── config.py                  # all hyperparameters / paths in one place
├── train.py                   # training entry point
├── eval.py                    # standalone verification evaluation
├── requirements.txt
├── models/
│   ├── iresnet.py              # IResNet50 / IResNet100 backbone
│   └── metrics.py               # ArcMarginProduct (ArcFace head)
├── datasets/
│   ├── casia_webface.py         # training set loader
│   └── verification.py          # LFW-style pair-verification loader
├── utils/
│   ├── utils.py                  # AverageMeter, checkpoint I/O, logging, seeding
│   └── lfw_eval.py                # 10-fold verification protocol
├── scripts/
│   └── check_dataset.py           # sanity-check a downloaded dataset folder
└── data/
    └── README.md                   # expected folder layout for your data
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for realistic training times (the paper trains on 8x
Tesla P40s); it will also run on CPU for testing with a tiny dataset, just
very slowly.

## Prepare your data

See `data/README.md` for the exact folder layout expected for CASIA-WebFace
and (optionally) LFW. Once your data is in place:

```bash
python scripts/check_dataset.py --root data/casia_webface_112x112
```

## Train

```bash
python train.py
```

Paper defaults (`config.py`): backbone `iresnet50`, batch size 512, SGD
lr=0.1 (linearly scaled if you change batch size), momentum 0.9, weight
decay 5e-4, 32 epochs, LR divided by 10 at epochs 20 and 28, ArcFace `s=64`,
`m=0.5`.

**On a single GPU**, batch size 512 usually won't fit — shrink it and let
the linear-scaling rule adjust the learning rate for you:

```bash
python train.py --batch_size 128          # lr becomes 0.1 * 128/512 = 0.025
```

Other common overrides:

```bash
python train.py --backbone iresnet100 --epochs 24
python train.py --resume checkpoints/epoch_10.pth
python train.py --no_amp                  # disable mixed precision
```

Checkpoints land in `checkpoints/epoch_<N>.pth`; logs go to
`logs/train.log` (and stdout). If `data/lfw_112x112` + `data/lfw_pairs.txt`
exist, verification accuracy is logged after every epoch (`val_interval` in
`config.py`).

## Evaluate a checkpoint

```bash
python eval.py --checkpoint checkpoints/epoch_31.pth
```

Point `--val_dir` / `--val_pairs` at CFP-FP, AgeDB, CALFW or CPLFW (in their
LFW-pairs-style redistributions) to reproduce the other columns of Table 2 /
Table 5 without any code changes.

## Expected results

Table 2 of the paper reports, for exactly this setting
(`[CASIA, ResNet50, ArcFace(m=0.5)]`):

| Dataset | Accuracy |
|---|---|
| LFW     | 99.53% |
| CFP-FP  | 95.56% |
| AgeDB   | 95.15% |

Use these as a sanity check once training finishes — if you land noticeably
below them, double check face alignment quality (misaligned crops are the
most common cause of underperformance), the learning-rate scaling if you
changed batch size, and that you trained the full 32 epochs.

## Notes / deviations from the paper

- **Mixed precision** (`use_amp=True` by default) isn't in the original
  paper (it predates widespread AMP use) but doesn't change the method —
  it's on by default purely for speed/memory on modern single-GPU setups.
  Disable with `--no_amp` for a closer match to the original fp32 setup.
- **Gradient clipping** (`grad_clip_norm=5.0`) is a defensive addition not
  in the paper; set it to `0` in `config.py` to turn it off.
- The paper trains on 8 GPUs with batch size 512 total; single-GPU training
  with a smaller batch size and linearly-scaled LR (handled automatically,
  see `Config.scaled_lr()`) is the standard way to approximate this, though
  it isn't guaranteed to match the paper's numbers exactly at very small
  batch sizes.
