"""
Sanity-check a CASIA-WebFace-style folder before training: counts
identities/images, flags images that aren't 112x112 (they'll still be
resized automatically at load time, this is just a heads-up), and tries to
open a sample of files to catch corrupt images early.

Usage:
    python scripts/check_dataset.py --root data/casia_webface_112x112
"""
import argparse
import os
import sys

from PIL import Image

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sample", type=int, default=300, help="number of images to open and verify")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Not a directory: {args.root}")

    identities = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    if not identities:
        sys.exit(f"No identity sub-folders found under {args.root}")

    per_identity_counts = []
    all_paths = []
    for identity in identities:
        identity_dir = os.path.join(args.root, identity)
        files = [f for f in os.listdir(identity_dir) if f.lower().endswith(IMG_EXTENSIONS)]
        per_identity_counts.append(len(files))
        all_paths.extend(os.path.join(identity_dir, f) for f in files)

    total_images = len(all_paths)
    empty_identities = sum(1 for c in per_identity_counts if c == 0)

    sample_paths = all_paths[:: max(1, total_images // max(args.sample, 1))][: args.sample]
    bad_files, wrong_size = [], 0
    for path in sample_paths:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if img.size != (112, 112):
                    wrong_size += 1
        except Exception as e:  # noqa: BLE001 - we want to report *any* read failure
            bad_files.append((path, str(e)))

    print(f"Root:                 {args.root}")
    print(f"Identities found:     {len(identities)} ({empty_identities} with 0 images)")
    print(f"Total images:         {total_images}")
    if per_identity_counts:
        print(f"Images / identity:    min={min(per_identity_counts)}, "
              f"max={max(per_identity_counts)}, avg={total_images / len(identities):.1f}")
    print(f"Sample checked:       {len(sample_paths)} images")
    print(f"  not 112x112:        {wrong_size} (will be auto-resized, just flagging)")
    print(f"  unreadable/corrupt: {len(bad_files)}")
    for path, err in bad_files[:10]:
        print(f"    [bad] {path}: {err}")
    if len(bad_files) > 10:
        print(f"    ... and {len(bad_files) - 10} more")


if __name__ == "__main__":
    main()
