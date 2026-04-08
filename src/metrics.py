"""Evaluation metrics for recommendations."""

import numpy as np


def hit_ratio_at_k(predictions, ground_truth, k):
    """Compute Hit Ratio @ K.

    Args:
        predictions: List of predicted item lists (per user)
        ground_truth: List of ground truth items (per user), can be string or list
        k: Cutoff position

    Returns:
        Hit Ratio @ K
    """
    hits = 0
    total = 0

    for pred, gt in zip(predictions, ground_truth):
        # Handle both string and list formats
        if isinstance(gt, list):
            gt_set = set(gt)
            pred_set = set(pred[:k])
            if len(gt_set & pred_set) > 0:
                hits += 1
        else:
            if gt in pred[:k]:
                hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def ndcg_at_k(predictions, ground_truth, k):
    """Compute NDCG @ K.

    Args:
        predictions: List of predicted item lists (per user)
        ground_truth: List of ground truth items (per user), can be string or list
        k: Cutoff position

    Returns:
        NDCG @ K
    """
    ndcgs = []

    for pred, gt in zip(predictions, ground_truth):
        # Handle both string and list formats
        if isinstance(gt, list):
            gt_set = set(gt)
            # Check if any ground truth item is in predictions
            found = False
            for i, item in enumerate(pred[:k]):
                if item in gt_set:
                    dcg = 1.0 / np.log2(i + 2)  # position starts at 0, so +2
                    idcg = 1.0 / np.log2(2)  # Ideal: first position
                    ndcgs.append(dcg / idcg)
                    found = True
                    break
            if not found:
                ndcgs.append(0.0)
        else:
            if gt in pred[:k]:
                pos = pred.index(gt) + 1
                dcg = 1.0 / np.log2(pos + 1)
                idcg = 1.0 / np.log2(2)  # Ideal: first position
                ndcgs.append(dcg / idcg)
            else:
                ndcgs.append(0.0)

    return np.mean(ndcgs) if ndcgs else 0.0


def recall_at_k(predictions, ground_truth, k):
    """Compute Recall @ K.

    Args:
        predictions: List of predicted item lists (per user)
        ground_truth: List of ground truth items (per user)
        k: Cutoff position

    Returns:
        Recall @ K
    """
    recalls = []

    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt) if isinstance(gt, list) else {gt}
        pred_set = set(pred[:k])

        if len(gt_set) > 0:
            recall = len(gt_set & pred_set) / len(gt_set)
            recalls.append(recall)
        else:
            recalls.append(0.0)

    return np.mean(recalls) if recalls else 0.0


def precision_at_k(predictions, ground_truth, k):
    """Compute Precision @ K.

    Args:
        predictions: List of predicted item lists (per user)
        ground_truth: List of ground truth items (per user)
        k: Cutoff position

    Returns:
        Precision @ K
    """
    precisions = []

    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt) if isinstance(gt, list) else {gt}
        pred_set = set(pred[:k])

        if k > 0:
            precision = len(gt_set & pred_set) / k
            precisions.append(precision)
        else:
            precisions.append(0.0)

    return np.mean(precisions) if precisions else 0.0


def compute_metrics(predictions, ground_truth, k_values):
    """Compute all metrics at multiple K values.

    Args:
        predictions: List of predicted item lists
        ground_truth: List of ground truth items
        k_values: List of K values

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    for k in k_values:
        metrics[f'hr@{k}'] = hit_ratio_at_k(predictions, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, ground_truth, k)
        metrics[f'recall@{k}'] = recall_at_k(predictions, ground_truth, k)
        metrics[f'precision@{k}'] = precision_at_k(predictions, ground_truth, k)

    return metrics
