"""
Standard 10-fold verification protocol (Huang et al., 2007) — the same
evaluation procedure the paper uses for its LFW / CFP-FP / AgeDB / CALFW /
CPLFW numbers in Table 2 and Table 5: split the pairs into 10 folds, pick
the distance threshold that maximizes accuracy on the other 9 folds, and
report accuracy on the held-out fold. The mean +/- std over the 10 folds is
the reported number.
"""
import numpy as np
import torch
from sklearn.model_selection import KFold


@torch.no_grad()
def compute_embeddings(backbone, data_loader, device):
    """Runs a VerificationPairs loader through the backbone and collects
    (embeddings1, embeddings2, labels) as numpy arrays."""
    backbone.eval()
    all_emb1, all_emb2, all_labels = [], [], []
    for img1, img2, is_same in data_loader:
        emb1 = backbone(img1.to(device))
        emb2 = backbone(img2.to(device))
        all_emb1.append(emb1.cpu().numpy())
        all_emb2.append(emb2.cpu().numpy())
        all_labels.append(np.asarray(is_same))
    return (
        np.concatenate(all_emb1, axis=0),
        np.concatenate(all_emb2, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def _accuracy_at_threshold(threshold, dist, actual_issame):
    predict_issame = dist < threshold
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    return (tp + tn) / dist.size


def evaluate_verification(embeddings1, embeddings2, actual_issame, num_folds=10, threshold_step=0.005):
    """
    Args:
        embeddings1, embeddings2: (N, D) arrays, the two sides of each pair.
        actual_issame: (N,) boolean array.
    Returns:
        (mean_accuracy, std_accuracy, mean_best_threshold)
    """
    actual_issame = np.asarray(actual_issame).astype(bool)
    assert embeddings1.shape[0] == embeddings2.shape[0] == actual_issame.shape[0]

    # cosine distance in [0, 2]: 0 for identical direction, 2 for opposite.
    e1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    e2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    dist = 1.0 - np.sum(e1 * e2, axis=1)

    thresholds = np.arange(0.0, 2.0, threshold_step)
    kfold = KFold(n_splits=num_folds, shuffle=False)
    indices = np.arange(len(actual_issame))

    accuracies = np.zeros(num_folds)
    best_thresholds = np.zeros(num_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        train_acc = np.array([
            _accuracy_at_threshold(t, dist[train_idx], actual_issame[train_idx])
            for t in thresholds
        ])
        best_threshold = thresholds[np.argmax(train_acc)]
        best_thresholds[fold_idx] = best_threshold
        accuracies[fold_idx] = _accuracy_at_threshold(best_threshold, dist[test_idx], actual_issame[test_idx])

    return float(accuracies.mean()), float(accuracies.std()), float(best_thresholds.mean())
