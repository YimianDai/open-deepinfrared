import os.path as osp
from collections import OrderedDict

import numpy as np
from skimage import measure as skm

import mmcv
from mmcv.utils import print_log
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose

from deepir.core.evaluation import eval_mnocoap
from deepir.datasets.pipelines import LoadBinaryAnnotations

@DATASETS.register_module()
class SIRSTSeg2NoCoDataset(CustomDataset):
    """Seg2NoCo on SIRST dataset. The file structure is as followed.

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
        split (str): Split txt file for SIRST dataset.
    """

    # CLASSES = ('background', 'target')
    # PALETTE = [[0, 0, 0], [255, 255, 255]]
    CLASSES = ('background', 'target')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 split,
                 gt_noco_map_loader_cfg=None,
                 noco_thrs=None,
                 img_suffix='.png',
                 seg_map_suffix='_pixels0.png',
                 **kwargs):
        super(SIRSTSeg2NoCoDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            split=split,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        self.gt_seg_map_loader = LoadBinaryAnnotations()
        if gt_noco_map_loader_cfg is None:
            gt_noco_map_loader_cfg=[
                dict(type='LoadImageFromFile', color_type='grayscale'),
                dict(type='LoadBinaryAnnotations'),
                dict(type='NoCoTargets')]
        self.gt_noco_map_loader = Compose(gt_noco_map_loader_cfg)
        if noco_thrs is None:
            noco_thrs = np.linspace(
                .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            noco_thrs = [noco_thrs] if isinstance(
                noco_thrs, float) else noco_thrs
        self.noco_thrs = noco_thrs
        self.best_mnocoap = -np.inf

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

    def eval_mean_nocoap(self,
                         results,
                         logger=None,
                         noco_thrs=None):
        """Evaluation on mNoCoAP.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            noco_thrs (Sequence[float], optional): NoCo threshold used for
                evaluating recalls/mNoCoAPs. If set to a list, the average of
                all NoCos will also be computed. If not specified, [0.1, 0.2,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] will be used.
                Default: None.

        Returns:
            dict
        """
        if noco_thrs is None:
            noco_thrs = np.linspace(
                .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            noco_thrs = [noco_thrs] if isinstance(
                noco_thrs, float) else noco_thrs

        # prepare inputs for eval_mnocoap
        det_centroids = [result[0] for result in results]
        gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
                        for i in range(len(self))]
        gt_bboxes = [self.get_gt_bbox_by_idx(i) for i in range(len(self))]

        eval_results = OrderedDict()
        mean_nocoaps = []
        for noco_thr in noco_thrs:
            print_log(f'\n{"-" * 15}noco_thr: {noco_thr}{"-" * 15}')
            mean_nocoap, _ = eval_mnocoap(det_centroids,
                                          gt_noco_maps,
                                          gt_bboxes,
                                          noco_thr=noco_thr,
                                          logger=logger)
            mean_nocoaps.append(mean_nocoap)
            eval_results[f'NoCoAP{int(noco_thr * 100):02d}'] = round(
                mean_nocoap, 3)
        eval_results['mNoCoAP'] = sum(mean_nocoaps) / len(mean_nocoaps)
        print("eval_results['mNoCoAP']:", eval_results['mNoCoAP'])
        if self.best_mnocoap < eval_results['mNoCoAP']:
            self.best_mnocoap = eval_results['mNoCoAP']
        print("best eval_results['mNoCoAP']:", self.best_mnocoap)
        print_log(f"\n best eval_results['mNoCoAP']: {self.best_mnocoap}")

        return eval_results

    def evaluate(self,
                 results,
                 metric='mNoCoAP',
                 logger=None,
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
        allowed_metrics = ['mNoCoAP',]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        elif metric == ['mNoCoAP']:
            eval_results = self.eval_mean_nocoap(results=results,
                                                 logger=logger,
                                                 noco_thrs=self.noco_thrs)
        else:
            raise ValueError(f"unsupported metric {metric}")
        return eval_results

