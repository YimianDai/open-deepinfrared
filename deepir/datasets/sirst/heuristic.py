import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
from prettytable import PrettyTable

import numpy as np
from skimage import measure as skm

import mmcv
from mmcv.utils import print_log

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose

from deepir.core.evaluation import eval_mnocoap
from deepir.datasets.pipelines import LoadBinaryAnnotations

@DATASETS.register_module()
class SIRSTHeuristicDataset(CustomDataset):
    """Normalized Contrast Dataset for SIRST-LTE project. The file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── sirst
        │   │   ├── images
        │   │   │   ├── backgrounds
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── targets
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   ├── mixed
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── annotations
        │   │   │   ├── bboxes
        │   │   │   │   ├── xxx.xml
        │   │   │   │   ├── yyy.xml
        │   │   │   │   ├── zzz.xml
        │   │   │   ├── masks
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   ├── splits

    Args:
        split (str): Split txt file for datasets in SIRST-LTE project.
    """

    CLASSES = ('background', 'target')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, score_thr=0.5,
                 gt_noco_map_loader_cfg=[
                     dict(type='LoadImageFromFile', color_type='grayscale'),
                     dict(type='LoadBinaryAnnotations'),
                     dict(type='NoCoTargets')],
                 **kwargs):
        super(SIRSTHeuristicDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_pixels0.png',
            split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        self.score_thr = score_thr
        self.gt_noco_map_loader = Compose(gt_noco_map_loader_cfg)
        self.gt_seg_map_loader = LoadBinaryAnnotations()

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

    def evaluate(self,
                 results,
                 metric='mNoCoAP',
                 logger=None,
                 noco_thrs=None):
        """Evaluate in NoCo protocol.

        Args:
            results (list[tuple(np.ndarray)] | list[np.ndarray]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mNoCoAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            noco_thrs (Sequence[float], optional): NoCo threshold used for
                evaluating recalls/mNoCoAPs. If set to a list, the average of
                all NoCos will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mNoCoAP. If not specified, all bounding boxes would be included
                in evaluation. Default: None.

        Returns:
            dict[str, float]: NoCoAP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mNoCoAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if noco_thrs is None:
            noco_thrs = np.linspace(
                .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)

        # prepare inputs for eval_mnocoap
        # det_centroids = [self.get_pred_centroids(pred, self.score_thr)
        #                  for pred in results]
        det_centroids = [result[0] for result in results]
        gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
                        for i in range(len(self))]
        gt_bboxes = [self.get_gt_bbox_by_idx(i) for i in range(len(self))]

        eval_results = OrderedDict()
        noco_thrs = [noco_thrs] if isinstance(noco_thrs, float) else noco_thrs
        if metric == 'mNoCoAP':
            assert isinstance(noco_thrs, np.ndarray),\
                'noco_thrs should be a ndarray'
            ds_name = self.CLASSES
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
        # TODO: `eval_recalls` is not supported yet
        return eval_results