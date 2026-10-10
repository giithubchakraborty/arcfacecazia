"""
Central configuration for training/evaluating ArcFace on CASIA-WebFace.

All hyperparameters below (batch size, LR schedule, s, m, optimizer) are
the ones reported for the [CASIA, ResNet50, ArcFace] setting in Section 4.1
and Table 2 of the paper. Edit this file (or override via CLI flags in
train.py) to fit your hardware.
"""
import os


class Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # Folder-per-identity layout, images already aligned/cropped to 112x112:
    #   train_dir/
    #       <identity_1>/*.jpg
    #       <identity_2>/*.jpg
    train_dir = "./data/casia_webface_112x112"

    # Verification sets, LFW-style (name + pairs.txt). CFP-FP / AgeDB / etc.
    # can reuse the same loader (datasets/verification.py) by pointing these
    # at their own image folder + pairs file.
    val_dir = "./data/lfw_112x112"
    val_pairs = "./data/lfw_pairs.txt"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    backbone = "iresnet50"       # "iresnet50" or "iresnet100"
    embedding_size = 512         # paper fixes the embedding dim d = 512
    dropout = 0.4

    # ------------------------------------------------------------------
    # ArcFace head (Eq. 3 in the paper)
    # ------------------------------------------------------------------
    s = 64.0                     # feature scale
    m = 0.5                      # additive angular margin (best value, Table 2)
    easy_margin = False

    # ------------------------------------------------------------------
    # Optimization — CASIA schedule from Section 4.1
    # ------------------------------------------------------------------
    batch_size = 512             # paper default (8 GPUs). Lower this on 1 GPU.
    base_lr = 0.1                # paper LR for batch_size=512
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 32
    lr_milestones = [20, 28]     # divide LR by 10 at these epochs
    lr_gamma = 0.1

    # If you shrink batch_size, linearly scale base_lr as well, e.g.
    #   lr = base_lr * (batch_size / 512)
    # train.py does this automatically — see `scaled_lr()` below.

    use_amp = True                # mixed precision (not in the original paper,
                                   # but safe/standard and much faster on 1 GPU)
    grad_clip_norm = 5.0           # 0 / None disables clipping

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    num_workers = 8
    seed = 42
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    resume = ""                   # path to a checkpoint to resume from
    log_interval = 50             # steps between console/log-file updates
    val_interval = 1              # run LFW-style verification every N epochs
    save_interval = 1             # checkpoint every N epochs

    def __init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def scaled_lr(self):
        """Linear LR scaling rule for batch sizes other than the paper's 512."""
        return self.base_lr * (self.batch_size / 512.0)
