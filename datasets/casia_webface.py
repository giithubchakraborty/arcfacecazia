"""
Dataset loader for CASIA-WebFace, expected as a folder-per-identity tree
of images already aligned/cropped to 112x112 (five-point alignment, as
described in Section 4.1 of the paper):

    casia_webface_112x112/
        0000045/
            001.jpg
            002.jpg
            ...
        0000099/
            001.jpg
            ...

Any redistribution of CASIA-WebFace pre-cropped to 112x112 in this layout
will work as-is. If your copy uses a different layout (e.g. an InsightFace
.rec/.idx record file), convert it to this folder structure first, or swap
this class for one that reads your format — everything downstream
(models, training loop, loss) only depends on `__getitem__` returning
(image_tensor, integer_label).
"""
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class CASIAWebFace(Dataset):
    def __init__(self, root, transform=None, input_size=112):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Dataset root not found: {root}\n"
                f"See data/README.md for the expected folder layout."
            )

        identities = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if not identities:
            raise RuntimeError(f"No identity sub-folders found under {root}")

        self.class_to_idx = {identity: idx for idx, identity in enumerate(identities)}
        self.samples = []
        for identity in identities:
            identity_dir = os.path.join(root, identity)
            label = self.class_to_idx[identity]
            for fname in sorted(os.listdir(identity_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((os.path.join(identity_dir, fname), label))

        if not self.samples:
            raise RuntimeError(f"Found {len(identities)} identity folders but 0 images under {root}")

        self.num_classes = len(identities)
        self.transform = transform or self.train_transform(input_size)

    @staticmethod
    def train_transform(input_size=112):
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # maps [0,1] -> ~[-1,1]
        ])

    @staticmethod
    def eval_transform(input_size=112):
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
