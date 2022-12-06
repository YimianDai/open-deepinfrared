import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
from prettytable import PrettyTable

import numpy as np
from skimage import measure as skm

import torch

import mmcv
from mmcv.utils import print_log
from mmseg.core import eval_metrics, pre_eval_to_metrics
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose

from deepir.core.evaluation import eval_mlocap_seg2cen, eval_mnocoap
from deepir.datasets.pipelines import LoadBinaryAnnotations


@DATASETS.register_module()
class SIRSTSegDataset(CustomDataset):
    """Segmentation on SIRST dataset.

    Args:
        split (str): Split txt file for SIRST dataset.
    """

    CLASSES = ('background', 'target')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split,
                 **kwargs):
        super(SIRSTSegDataset, self).__init__(img_suffix='.png',
                                              seg_map_suffix='_pixels0.png',
                                              split=split,
                                              **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

        self.annotations = []
        self.centerness_thrs = [centerness_thr] if isinstance(centerness_thr,
            float) else centerness_thr
        self.score_thr = score_thr
        self.gt_seg_map_loader = LoadBinaryAnnotations()
        self.gt_noco_map_loader = Compose(gt_noco_map_loader_cfg)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_map = gt_seg_map / 255
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_pred_centroids(self, pred, score_thr=0.5):
        """Convert pred noco_map to centroid detection results
        Args:
            pred (np.ndarray): shape (1, H, W)

        Returns:
            det_centroids (np.ndarray): shape (num_dets, 3)
        """

        if pred.ndim == 3:
            pred = pred.squeeze(0)
        seg_mask = (pred > score_thr).astype(int)
        gt_labels = skm.label(seg_mask, background=0)
        gt_regions = skm.regionprops(gt_labels)
        centroids = []
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            tgt_pred = pred[ymin:ymax, xmin:xmax]
            ridx, cind = np.unravel_index(np.argmax(tgt_pred, axis=None),
                                          tgt_pred.shape)
            tgt_score = tgt_pred[ridx, cind]
            centroids.append((xmin + cind, ymin + ridx, tgt_score))
            # centroids.append((xmin + cind, ymin + ridx))
        if len(centroids) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        else:
            return np.array(centroids, dtype=np.float32)

    def get_gt_bbox_by_idx(self, index):
        """Get gt bboxes for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        gt_semantic_seg = results['gt_semantic_seg']
        label_img = skm.label(gt_semantic_seg, background=0)
        regions = skm.regionprops(label_img)
        bboxes = []
        for region in regions:
            ymin, xmin, ymax, xmax = region.bbox
            bboxes.append([xmin, ymin, xmax, ymax])
        if len(bboxes) == 0:
            return np.zeros((0, 4))
        else:
            return np.array(bboxes)

    def get_gt_noco_map_by_idx(self, index):
        """Get one ground truth normalized contrast map for evaluation."""
        img_info = self.img_infos[index]
        ann_info = self.get_ann_info(index)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_noco_map_loader(results)
        return results['gt_noco_map']

    def seg_map_to_det_anno(self, gt_seg_map):
        """Convert gt_seg_map to bboxes and labels
        Args:
            gt_seg_map (np.ndarray): shape (H, W)

        Returns:
            ann (dict): BBox annotation info of given gt_seg_map. Keys of 
                annotations are:

                - `bboxes`: numpy array of shape (n, 4)
                - `labels`: numpy array of shape (n, )
                - `bboxes_ignore` (optional): numpy array of shape (k, 4)
                - `labels_ignore` (optional): numpy array of shape (k, )
        """
        bboxes = []
        labels = []
        label_map = skm.label(gt_seg_map, background=0)
        regions = skm.regionprops(label_map)
        for props in regions:
            # props.bbox: (min_row, min_col, max_row, max_col)
            ymin, xmin, ymax, xmax = props.bbox
            # ymax -= 1 # ymax that props.bbox returns is not labeled as 1
            # xmax -= 1 # xmax that props.bbox returns is not labeled as 1
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(gt_seg_map[props.coords[0, 0], props.coords[0, 1]])
        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            labels = np.array(labels)
        else:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, 4))
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        extended_allowed_metrics = ['mLocAP', 'mNoCoAP', *allowed_metrics]
        if not set(metric).issubset(set(extended_allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        if metric == ['mLocAP']:
            eval_results = self._evaluate_centroids(
                results=results,
                metric=metric[0],
                logger=logger,
                centerness_thr=self.centerness_thrs,
                scale_ranges=None)
        elif metric == ['mNoCoAP']:
            noco_thrs = np.linspace(
                .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            noco_thrs = [noco_thrs] if isinstance(noco_thrs,
                                                  float) else noco_thrs
            eval_results = self._eval_mean_nocoap(
                results=results,
                logger=logger,
                noco_thrs=noco_thrs,
                score_thr=self.score_thr)
        elif set(metric).issubset(set(allowed_metrics)):
            eval_results = self._evaluate_seg_maps(
                results=results,
                metric=metric,
                logger=logger,
                efficient_test=efficient_test)
        else:
            raise ValueError(
                f"unsupported metric {metric}")
        return eval_results

    def _evaluate_seg_maps(self,
                           results,
                           metric='mIoU',
                           logger=None,
                           gt_seg_maps=None,
                           **kwargs):
        # TODO: 要将 results 从 sigmoid 的输出转化成 0 / 1 的 seg_pred
        results = [(seg_pred > self.score_thr).long().squeeze(1)
                   for seg_pred in results]
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

    def _evaluate_centroids(self,
                            results,
                            metric='mLocAP',
                            logger=None,
                            centerness_thr=[0.1, 0.3, 0.5, 0.7, 0.9],
                            scale_ranges=None):
        if not self.annotations:
            gt_seg_maps = self.get_gt_seg_maps(efficient_test=False)
            self.annotations = [self.seg_map_to_det_anno(seg_map) 
                for seg_map in gt_seg_maps]
        eval_results = OrderedDict()
        centerness_thrs = [centerness_thr] if isinstance(centerness_thr,
            float) else centerness_thr

        # start to calculate mLocAP
        assert isinstance(centerness_thrs, list)
        mean_locaps = []
        for centerness_thr in centerness_thrs:
            print_log(f'\n{"-" * 15}iou_thr: {centerness_thr}{"-" * 15}')
            mean_locap, _ = eval_mlocap_seg2cen(
                results,
                self.annotations,
                scale_ranges=None,
                centerness_thr=centerness_thr,
                dataset=None,
                logger=logger)
            mean_locaps.append(mean_locap)
            eval_results[f'LocAP{int(centerness_thr * 100):02d}'] = \
                round(mean_locap, 3)
        eval_results['mLocAP'] = sum(mean_locaps) / len(mean_locaps)
        print("eval_results['mLocAP']:", eval_results['mLocAP'])
        return eval_results

    def _eval_mean_nocoap(self,
                          results,
                          logger=None,
                          noco_thrs=None,
                          score_thr=0.5):

        # prepare inputs for eval_mnocoap
        det_centroids = [self.get_pred_centroids(pred, score_thr)
                        for pred in results]
        gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
                        for i in range(len(self))]
        gt_bboxes = [self.get_gt_bbox_by_idx(i) for i in range(len(self))]

        eval_results = OrderedDict()
        mean_nocoaps = []
        for noco_thr in noco_thrs:
            print_log(f'\n{"-" * 15}noco_thr: {noco_thr}{"-" * 15}')
            mean_nocoap, _ = eval_mnocoap(
                det_centroids,
                gt_noco_maps,
                gt_bboxes,
                noco_thr=noco_thr,
                logger=logger)
            mean_nocoaps.append(mean_nocoap)
            eval_results[f'NoCoAP{int(noco_thr * 100):02d}'] = round(
                mean_nocoap, 3)
        eval_results['mNoCoAP'] = sum(mean_nocoaps) / len(mean_nocoaps)
        print("eval_results['mNoCoAP']:", eval_results['mNoCoAP'])
        # TODO: `eval_recalls` is not supported yet
        return eval_results