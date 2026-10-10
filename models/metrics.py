"""
ArcFace margin head.

Implements Eq. 2-3 of the paper:

    L2 = -log( e^{s*cos(theta_yi)} / (e^{s*cos(theta_yi)} + sum_{j!=yi} e^{s*cos(theta_j)}) )
    L3 = -log( e^{s*cos(theta_yi+m)} / (e^{s*cos(theta_yi+m)} + sum_{j!=yi} e^{s*cos(theta_j)}) )

i.e.: normalize both the embedding and the class-center weight vectors so the
logit is a pure cosine similarity, add an additive angular margin m to the
*target* class's angle, rescale by s, and let a standard softmax
cross-entropy do the rest. ArcMarginProduct below returns the rescaled
logits; combine it with nn.CrossEntropyLoss (see train.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ArcMarginProduct"]


class ArcMarginProduct(nn.Module):
    """
    Args:
        in_features: embedding dimension (512 in the paper).
        out_features: number of identities/classes N.
        s: feature scale (paper default 64, see Table 2).
        m: additive angular margin in radians (paper default 0.5, Table 2's
           best-performing setting on CASIA/ResNet50).
        easy_margin: if True, use the looser "cos(theta) > 0" fallback
            instead of the paper's exact monotonic threshold. Leave False to
            match the paper.
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Beyond pi - m, cos(theta + m) stops being monotonically decreasing
        # in theta, which would let the "margin" start helping a badly-
        # misclassified sample instead of penalizing it. Past this
        # threshold we fall back to a linear penalty (cos(theta) - mm) that
        # preserves monotonicity. See the ArcFace paper/official
        # implementation notes for this detail.
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # l2-normalize features and class-center weights (Eq. 2)
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # guard acos/sqrt domain
        sine = torch.sqrt(1.0 - cosine.pow(2))

        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s
        return logits


if __name__ == "__main__":
    # Quick sanity check: `python -m models.metrics`
    torch.manual_seed(0)
    head = ArcMarginProduct(in_features=8, out_features=5, s=64.0, m=0.5)
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(embeddings, labels)
    print("logits shape:", tuple(logits.shape))
    print("any NaN:", torch.isnan(logits).any().item())
