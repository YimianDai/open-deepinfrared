from multiprocessing import Pool
import numpy as np

from mmdet.core import average_precision
from mmdet.core import print_map_summary


def darts_scoring(pred_centroids, gt_bboxes):
    """Calculate the normalized centerness between each predicted centroid and each gt_bbox.

    Args:
        pred_centroids(ndarray): shape (n, 2), (x_c, y_c)
        gt_bboxes(ndarray): shape (k, 4), (tl_x, tl_y, br_x, br_y)

    Returns:
        centernesses (ndarray): shape (n, k)
    """

    # 1. 是否在 BBox 里面, 在的话, detection score 就为 1, 不在就为 0;
    #     + x 是否满足 > x_beg, < x_end, y 是否满足 > y_beg, < y_end;
    #     + 统统满足就是 1; 否则就是 0;
    # 2. 如果在, 则可以按照 FCOS 的 centerness 来计算 centerness;
    # 3. pred_centroids 的 x 与 gt_bboxes 的 x_beg 和 x_end 相减, y 也类似炮制, 可以获得 (l, t, r, b), 从而可以计算出 centernesses;
    # 4. detection score 和 centernesses 相乘, 才是最终的 centernesses;

    pred_centroids = pred_centroids.astype(np.float32)
    gt_bboxes = gt_bboxes.astype(np.float32)
    rows = pred_centroids.shape[0]
    cols = gt_bboxes.shape[0]
    # score on darts
    sods = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return sods
    gt_xls = gt_bboxes[:, 0]
    gt_xrs = gt_bboxes[:, 2]
    gt_yts = gt_bboxes[:, 1]
    gt_ybs = gt_bboxes[:, 3]
    for i in range(pred_centroids.shape[0]):
        xc, yc = pred_centroids[i, 0], pred_centroids[i, 1]
        det_x_flags = (xc > gt_xls) & (xc < gt_xrs)
        det_y_flags = (yc > gt_yts) & (yc < gt_ybs)
        det_flags = det_x_flags & det_y_flags
        if np.any(det_flags):
            # this centroid hits at least one gt bbox
            inds = np.where(det_flags)
            ls = xc - gt_xls[inds]
            ts = yc - gt_yts[inds]
            rs = gt_xrs[inds] - xc
            bs = gt_ybs[inds] - yc
            min_l_r = np.minimum(ls, rs)
            min_t_b = np.minimum(ts, bs)
            max_l_r = np.maximum(ls, rs)
            max_t_b = np.maximum(ts, bs)
            centerness = np.sqrt(min_l_r * min_t_b / (max_l_r * max_t_b))
            sods[i, inds] = centerness
    return sods

def tpfp_centerness(det_centroids,
                    gt_bboxes,
                    gt_bboxes_ignore=None,
                    centerness_thr=0.25,
                    area_ranges=None):
    """Check if predicted centroids are true positive or false positive.

    Args:
        det_centroids (ndarray): Detected object centroids of this image, 
            of shape (m, 3).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        centerness_thr (float): centerness threshold to be considered as 
            matched. Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate((
        np.zeros(gt_bboxes.shape[0], dtype=np.bool),
        np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_dets = det_centroids.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        return tp, fp

    centernesses = darts_scoring(det_centroids, gt_bboxes)
    # for each det, the max centerness with all gts
    centernesses_max = centernesses.max(axis=1)
    # for each det, which gt overlaps most with it
    centernesses_argmax = centernesses.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_centroids[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if centernesses_max[i] >= centerness_thr:
                matched_gt = centernesses_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
            else:
                fp[k, i] = 1
    return tp, fp

def get_cls_results_seg2cen(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected centroids, gt bboxes, ignored gt bboxes
    """

    cls_dets = [img_res[class_id-1] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann['labels']) > 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
        else:
            cls_gts.append(np.empty((0, 4), dtype=np.float32))

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore

def eval_mlocap_seg2cen(det_results,
                        annotations,
                        scale_ranges=None,
                        centerness_thr=0.5,
                        dataset=None,
                        logger=None,
                        tpfp_fn=None,
                        nproc=4):

    """Evaluate mLocAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        centerness_thr (float): Centerness threshold to be considered as 
            matched. Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mLocAP, [dict, dict, ...])
    """

    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num

    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = []
    for i in range(1, num_classes+1):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results_seg2cen(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        tpfp_fn = tpfp_centerness
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_fn,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [centerness_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (
                    bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((
                        gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()

        ap = average_precision(recalls, precisions, mode='area')
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def get_cls_results_det2cen(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected centroids, gt bboxes, ignored gt bboxes
    """

    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))
    return cls_dets, cls_gts, cls_gts_ignore

def eval_mlocap_det2cen(det_results,
                        annotations,
                        scale_ranges=None,
                        centerness_thr=0.5,
                        dataset=None,
                        logger=None,
                        tpfp_fn=None,
                        nproc=4):

    """Evaluate mLocAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `centroids`: numpy array of shape (n, 2)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        centerness_thr (float): Centerness threshold to be considered as 
            matched. Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mLocAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num

    area_ranges = (
        [(rg[0]**2, rg[1]**2) for rg in scale_ranges]
        if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = []
    for i in range(0, num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results_det2cen(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        tpfp_fn = tpfp_centerness
        if not callable(tpfp_fn):
            raise ValueError(
                f'tpfp_fn has to be a function or None, but got {tpfp_fn}')

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_fn,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [centerness_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0]) * (
                    bbox[:, 3] - bbox[:, 1])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((
                        gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        ap = average_precision(recalls, precisions, mode='area')
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

