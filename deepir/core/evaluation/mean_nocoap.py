from multiprocessing import Pool
import time
import numpy as np
from skimage import measure as skm

from mmdet.core.evaluation import average_precision, print_map_summary


def contain_point(cx, cy, bbox):
    """Check if a point is inside a bbox.

    Args:
        cx (float): x coordinate of the point.
        cy (float): y coordinate of the point.
        bbox (ndarray): bounding box of shape (4,).

    Returns:
        bool: Whether the point is inside the bbox.
    """
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def get_matched_gt_bbox_idx(cx, cy, gt_bboxes):
    """Get matched gt bbox index.

    Returns:
        int: Matched gt bbox index.
    """
    matched_gt_bbox_idx = -1
    for i, bbox in enumerate(gt_bboxes):
        if contain_point(cx, cy, bbox):
            matched_gt_bbox_idx = i
            break
    return matched_gt_bbox_idx


def tpfp_noco(det_centroids, gt_noco_map, gt_bboxes, noco_thr):
    """Check if detected centroids are true positive or false positive.

    Args:
        det_centroids (ndarray): Detected centroids of this image,
            of shape (m, 3).
        gt_noco_map (ndarray): GT noco maps of this image, of shape (H, W).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        noco_thr (float): NoCo threshold to be considered as matched.
            Default: 0.5.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_dets,).
    """
    num_dets = det_centroids.shape[0]
    num_gts = gt_bboxes.shape[0]

    tp = np.zeros(num_dets, dtype=np.float32)
    fp = np.zeros(num_dets, dtype=np.float32)

    # if there is no gt bboxes in this image, then all det centroids
    # are false positives
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp

    img_hei, img_wid = gt_noco_map.shape
    gt_covered = np.zeros(num_gts, dtype=bool)
    sort_inds = np.argsort(-det_centroids[:, -1])
    for i in sort_inds:
        cx, cy = det_centroids[i, :2]
        crow, ccol = int(np.round(cy)), int(np.round(cx))
        if crow == img_hei:
            crow = img_hei - 1
        if img_wid == ccol:
            ccol = img_wid - 1
        noco_val = gt_noco_map[crow, ccol]
        if noco_val > noco_thr:
            # get matched gt bbox
            matched_gt = get_matched_gt_bbox_idx(cx, cy, gt_bboxes)
            if matched_gt >= 0 and not gt_covered[matched_gt]:
                tp[i] = 1
                gt_covered[matched_gt] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def eval_mnocoap(det_centroids,
                 gt_noco_maps,
                 gt_bboxes,
                 noco_thr=0.5,
                 logger=None,
                 nproc=4):
    """Evaluate mNoCoAP of a dataset.

    Args:

        det_centroids (list[np.ndarray]): each row (cenx, ceny, score).
        gt_noco_maps (list[np.ndarray]): Ground truth normalized contrast map.
            The shape of each np.ndarray is (H, W).
        gt_bboxes (list[np.ndarray]): Ground truth bounding boxes.
        noco_thr (float): NoCo threshold to be considered as matched.
            Default: 0.5.
        score_thr (float): Score threshold to be considered as foregroundss.
            Default: 0.5.
        logger (logging.Logger | str | None): The way to print the mNoCoAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_noco` is used as default.
            Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple: (mNoCoAP, [dict, dict, ...])
    """
    assert len(det_centroids) == len(gt_noco_maps) == len(gt_bboxes)
    num_imgs = len(det_centroids)

    pool = Pool(nproc)
    eval_results = []

    # compute tp and fp for each image with multiple processes
    tpfp = pool.starmap(
        tpfp_noco, zip(
            det_centroids,
            gt_noco_maps,
            gt_bboxes,
            [noco_thr for _ in range(num_imgs)]))
    tp, fp = tuple(zip(*tpfp))
    # calculate gt bbox number in total
    num_gts = 0
    for bbox in gt_bboxes:
        num_gts += bbox.shape[0]
    # sort all det bboxes by score, also sort tp and fp
    det_centroids = np.vstack(det_centroids)
    num_dets = det_centroids.shape[0]
    sort_inds = np.argsort(-det_centroids[:, -1])
    tp = np.hstack(tp)[sort_inds]
    fp = np.hstack(fp)[sort_inds]
    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(num_gts, eps)
    precisions = tp / np.maximum((tp + fp), eps)
    ap = average_precision(recalls, precisions, mode='area')
    eval_results.append({
        'num_gts': num_gts,
        'num_dets': num_dets,
        'recall': recalls,
        'precision': precisions,
        'ap': ap
    })
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset=None, scale_ranges=None, logger=logger)

    return mean_ap, eval_results
