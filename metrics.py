
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def remap_labels(mask):
    """Remap labels to contiguous integers starting from 1."""
    mask = np.asarray(mask)
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if ids.size == 0:
        return mask
    mapping = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    mapping[ids] = np.arange(1, ids.size + 1)
    return mapping[mask]


def contingency_table(x, y):
    """Fast overlap table for label maps."""
    x = np.asarray(x, dtype=np.int64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    y_max = int(y.max()) if y.size > 0 else 0
    pair = x * (y_max + 1) + y
    counts = np.bincount(pair, minlength=(int(x.max()) + 1) * (y_max + 1))
    return counts.reshape((int(x.max()) + 1, y_max + 1)).astype(np.uint64)


def iou_matrix(true, pred):
    """IoU matrix between all instance pairs."""
    overlap = contingency_table(true, pred)
    n_pred = np.sum(overlap, axis=0, keepdims=True)
    n_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pred + n_true - overlap + 1e-6)
    iou[np.isnan(iou)] = 0
    return iou


def hungarian_match(iou, threshold=0.5):
    """Match predictions to ground truth using Hungarian algorithm."""
    n = min(iou.shape[0], iou.shape[1])
    cost = -(iou >= threshold).astype(float) - iou / (2 * n)
    t_idx, p_idx = linear_sum_assignment(cost)
    matched = iou[t_idx, p_idx] >= threshold
    return int(matched.sum())


def f1_score(true, pred, threshold=0.5):
    """F1 score using Hungarian matching."""
    true = remap_labels(true)
    pred = remap_labels(pred)

    n_gt = len(np.unique(true)) - 1
    n_pred = len(np.unique(pred)) - 1

    if n_pred == 0:
        return 0.0

    iou_mat = iou_matrix(true, pred)[1:, 1:]
    tp = hungarian_match(iou_mat, threshold)
    fp = n_pred - tp
    fn = n_gt - tp

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def dice_score(true, pred):
    """Binary Dice score."""
    true_bin = (true > 0).astype(int)
    pred_bin = (pred > 0).astype(int)
    inter = np.count_nonzero(true_bin & pred_bin)
    total = np.count_nonzero(true_bin) + np.count_nonzero(pred_bin)
    return 2 * inter / total if total > 0 else 0.0


def aji_score(true, pred):
    """Aggregated Jaccard Index."""
    true = remap_labels(true)
    pred = remap_labels(pred)

    overlap = contingency_table(true, pred)
    n_true, n_pred = overlap.shape[0] - 1, overlap.shape[1] - 1

    if n_true == 0 or n_pred == 0:
        return 0.0

    inter = overlap[1:, 1:].astype(float)
    true_area = overlap[1:, :].sum(axis=1)
    pred_area = overlap[:, 1:].sum(axis=0)
    union = true_area[:, None] + pred_area[None, :] - inter

    iou_mat = inter / (union + 1e-6)
    paired_pred = np.argmax(iou_mat, axis=1)
    max_iou = iou_mat.max(axis=1)

    valid = max_iou > 0
    paired_true = np.where(valid)[0]

    overall_inter = inter[paired_true, paired_pred[paired_true]].sum()
    overall_union = union[paired_true, paired_pred[paired_true]].sum()
    overall_union += true_area[~valid].sum() + pred_area[~np.isin(np.arange(n_pred), paired_pred[valid])].sum()

    return overall_inter / overall_union if overall_union > 0 else 0.0


def mask_to_boundary(mask, dilation_ratio=0.01):
    """Convert mask to boundary using morphological operations."""
    h, w = mask.shape
    mask = mask.astype(np.uint8)
    diag = np.sqrt(h**2 + w**2)
    dil = max(1, int(round(dilation_ratio * diag)))

    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), dtype=np.uint8), iterations=dil)
    eroded = eroded[1:h+1, 1:w+1]  # slice back to (h, w)
    return mask - eroded


def boundary_pq(true, pred, match_iou=0.5, dilation_ratio=0.01):
    """Boundary PQ (Panoptic Quality) score."""
    true = remap_labels(true)
    pred = remap_labels(pred)

    overlap = contingency_table(true, pred)
    n_true, n_pred = overlap.shape[0] - 1, overlap.shape[1] - 1

    if n_true == 0 or n_pred == 0:
        return {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}

    true_bounds = [mask_to_boundary((true == i).astype(np.uint8), dilation_ratio) for i in range(1, n_true + 1)]
    pred_bounds = [mask_to_boundary((pred == i).astype(np.uint8), dilation_ratio) for i in range(1, n_pred + 1)]

    true_areas = [b.sum() for b in true_bounds]
    pred_areas = [b.sum() for b in pred_bounds]

    pairs = np.argwhere(overlap[1:, 1:] > 0)
    iou_mat = np.zeros((n_true, n_pred))
    for i, j in pairs:
        inter = np.count_nonzero(true_bounds[i] & pred_bounds[j])
        union = true_areas[i] + pred_areas[j] - inter
        iou_mat[i, j] = inter / union if union > 0 else 0

    if match_iou >= 0.5:
        iou_mat[iou_mat <= match_iou] = 0
        t_idx, p_idx = np.nonzero(iou_mat)
    else:
        t_idx, p_idx = linear_sum_assignment(-iou_mat)
        valid = iou_mat[t_idx, p_idx] > match_iou
        t_idx, p_idx = t_idx[valid], p_idx[valid]

    tp = len(t_idx)
    sq = iou_mat[t_idx, p_idx].sum() / (tp + 1e-6)
    rq = tp / (tp + 0.5 * (n_pred - tp) + 0.5 * (n_true - tp))
    pq = sq * rq

    return {'pq': pq, 'sq': sq, 'rq': rq}



