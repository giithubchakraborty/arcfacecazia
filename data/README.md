# Data layout

This folder is empty on purpose — put your own data here (or symlink it),
following the layouts below. Nothing is downloaded automatically: CASIA-WebFace
and LFW are distributed for research use under their own terms, so fetch them
yourself and make sure your use complies with those terms.

## 1. Training set — CASIA-WebFace (112x112, folder-per-identity)

```
data/casia_webface_112x112/
    0000045/
        001.jpg
        002.jpg
        ...
    0000099/
        001.jpg
        ...
```

- One sub-folder per identity; the folder name becomes the class label
  (any string works, they're just sorted and indexed).
- Images should already be aligned and cropped to 112x112 (five-point
  alignment, as in Section 4.1 of the paper). If yours are a different size
  they'll be auto-resized to 112x112 at load time, but for best results
  align them properly first (e.g. with RetinaFace + similarity transform,
  as the paper does) rather than relying on a plain resize.
- Once your files are in place, sanity-check them before a long training
  run:
  ```
  python scripts/check_dataset.py --root data/casia_webface_112x112
  ```

## 2. Verification set — LFW (for monitoring / final accuracy)

```
data/lfw_112x112/
    Aaron_Peirsol/
        Aaron_Peirsol_0001.jpg
        Aaron_Peirsol_0002.jpg
    Aaron_Sorkin/
        Aaron_Sorkin_0001.jpg
    ...
data/lfw_pairs.txt
```

- `lfw_pairs.txt` is the standard LFW View 2 pairs file (6,000 pairs, 10
  folds): first line `10 300`, then lines `name idx1 idx2` (same person) or
  `name1 idx1 name2 idx2` (different people).
- This is the exact format used for LFW, and — in their commonly
  redistributed forms — for CFP-FP, AgeDB, CALFW and CPLFW too. To evaluate
  on any of those instead, just point `val_dir` / `val_pairs` (in
  `config.py`, or via `--val_dir` / `--val_pairs` on `eval.py`) at their
  image folder and pairs file — no code changes needed.

If you skip this set entirely, training still runs fine; you just won't get
the periodic verification-accuracy logs, and you can always add it later and
run `eval.py` against a saved checkpoint.
