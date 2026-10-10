"""
Pair-verification dataset for the LFW-style `pairs.txt` format used by LFW
itself and, in their commonly redistributed forms, by CFP-FP, AgeDB, CALFW
and CPLFW too (Table 2 / Section 4.1 of the paper) — so this one loader
covers all of them, just point `image_dir` / `pairs_path` at the right files.

Expected pairs.txt format (standard LFW View 2 layout):
    - optional first line: "<num_folds> <pairs_per_fold>"
    - matching pair:      "<name> <idx1> <idx2>"
    - mismatching pair:   "<name1> <idx1> <name2> <idx2>"

Expected image layout:
    image_dir/<name>/<name>_<idx zero-padded to 4 digits>.jpg
which is the standard LFW folder layout. Images are expected to already be
aligned/cropped to 112x112 as in the rest of this project.
"""
import os

from PIL import Image
from torch.utils.data import Dataset

from datasets.casia_webface import CASIAWebFace, IMG_EXTENSIONS


class VerificationPairs(Dataset):
    def __init__(self, image_dir, pairs_path, transform=None, input_size=112):
        self.image_dir = image_dir
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Verification image dir not found: {image_dir}")
        if not os.path.isfile(pairs_path):
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        self.transform = transform or CASIAWebFace.eval_transform(input_size)
        self.pairs = self._read_pairs(pairs_path)

    def _resolve_path(self, name, idx):
        stem = os.path.join(self.image_dir, name, f"{name}_{int(idx):04d}")
        for ext in IMG_EXTENSIONS:
            candidate = stem + ext
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No image found for {stem}.*")

    def _read_pairs(self, pairs_path):
        with open(pairs_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Skip an optional "<num_folds> <pairs_per_fold>" header line.
        if lines and len(lines[0].split()) == 2:
            lines = lines[1:]

        pairs = []
        skipped = 0
        for line in lines:
            parts = line.split()
            try:
                if len(parts) == 3:
                    name, idx1, idx2 = parts
                    path1 = self._resolve_path(name, idx1)
                    path2 = self._resolve_path(name, idx2)
                    is_same = True
                elif len(parts) == 4:
                    name1, idx1, name2, idx2 = parts
                    path1 = self._resolve_path(name1, idx1)
                    path2 = self._resolve_path(name2, idx2)
                    is_same = False
                else:
                    skipped += 1
                    continue
            except FileNotFoundError:
                skipped += 1
                continue
            pairs.append((path1, path2, is_same))

        if not pairs:
            raise RuntimeError(f"Parsed 0 usable pairs from {pairs_path}")
        if skipped:
            print(f"[VerificationPairs] skipped {skipped} pair(s) with missing images/bad format")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path1, path2, is_same = self.pairs[index]
        img1 = self.transform(Image.open(path1).convert("RGB"))
        img2 = self.transform(Image.open(path2).convert("RGB"))
        return img1, img2, is_same
