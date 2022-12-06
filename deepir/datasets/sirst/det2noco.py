import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

from PIL import Image
import numpy as np

import mmcv
from mmcv.utils import print_log
from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from deepir.core.centroid import det_results_to_noco_centroids, det_results_to_noco_peaks
from deepir.core.evaluation import eval_mnocoap

@DATASETS.register_module()
class SIRSTDet2NoCoDataset(CustomDataset):
    """SIRST dataset for bbox detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: mixed.
        ann_subdir (str): Subdir where annotations are. Default: annotations/bboxes.
    """
    CLASSES = ('Target', )

    def __init__(self,
                 min_size=None,
                 img_subdir='mixed',
                 ann_subdir='annotations/bboxes',
                 gt_noco_map_loader_cfg=None,
                 noco_thrs=None,
                 noco_mode='det2noco',
                 **kwargs):
        assert self.CLASSES or kwargs.get(
            'classes', None), 'CLASSES in `XMLDataset` can not be None.'
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        super(SIRSTDet2NoCoDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        if gt_noco_map_loader_cfg is None:
            gt_noco_map_loader_cfg=[
                dict(type='LoadImageFromFile', color_type='grayscale'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='NoCoTargets', mode='det2noco')]
        self.gt_noco_map_loader = Compose(gt_noco_map_loader_cfg)
        if noco_thrs is None:
            noco_thrs = np.linspace(
                .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            # noco_thrs = np.array([0.5])
            noco_thrs = [noco_thrs] if isinstance(
                noco_thrs, float) else noco_thrs
        self.noco_thrs = noco_thrs
        assert noco_mode in ['det2noco', 'noco_peak']
        self.noco_mode = noco_mode
        self.best_mnocoap = -np.inf

        self.gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
                            for i in range(len(self))]
        self.gt_bboxes = [self.get_ann_info(i)['bboxes']
                          for i in range(len(self))]

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join(self.img_subdir, f'{img_id}.png')
            xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))

        return data_infos

    def _filter_imgs(self, min_size=0):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            if bnd_box is not None:
                # TODO: check whether it is necessary to use int
                # Coordinates may be float type
                bbox = [
                    int(float(bnd_box.find('xmin').text)),
                    int(float(bnd_box.find('ymin').text)),
                    int(float(bnd_box.find('xmax').text)),
                    int(float(bnd_box.find('ymax').text))
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids

    def get_gt_noco_map_by_idx(self, index):
        """Get one ground truth normalized contrast map for evaluation."""
        img_info = self.data_infos[index]
        ann_info = self.get_ann_info(index)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_noco_map_loader(results)
        return results['gt_noco_map']

    def evaluate(self,
                 results,
                 metric='mNoCoAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in SIRST's NoCo protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mNoCoAP'.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: NoCoAP/AP/recall metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mNoCoAP',]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        if metric == ['mNoCoAP']:
            # if noco_thrs is None:
            #     noco_thrs = np.linspace(
            #         .1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)
            #     noco_thrs = [noco_thrs] if isinstance(
            #         noco_thrs, float) else noco_thrs

            # prepare inputs for eval_mnocoap
            # if self.noco_mode == 'det2noco':
            det_centroids = det_results_to_noco_centroids(results)
            # else:
            #     det_centroids = det_results_to_noco_peaks(results)
            # gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
            #                 for i in range(len(self))]
            # gt_bboxes = [self.get_ann_info(i)['bboxes']
            #              for i in range(len(self))]

            # compute mNoCoAP
            eval_results = OrderedDict()
            mean_nocoaps = []
            for noco_thr in self.noco_thrs:
                print_log(f'\n{"-" * 15}noco_thr: {noco_thr}{"-" * 15}')
                mean_nocoap, _ = eval_mnocoap(
                    det_centroids, self.gt_noco_maps, self.gt_bboxes,
                    noco_thr=noco_thr, logger=logger)
                # mean_nocoap, _ = eval_mnocoap(
                #     det_centroids, gt_noco_maps, gt_bboxes,
                #     noco_thr=noco_thr, logger=logger)
                mean_nocoaps.append(mean_nocoap)
                eval_results[f'NoCoAP{int(noco_thr * 100):02d}'] = round(
                    mean_nocoap, 3)
            eval_results['mNoCoAP'] = sum(mean_nocoaps) / len(mean_nocoaps)
            print("current eval_results['mNoCoAP']:", eval_results['mNoCoAP'])
            if self.best_mnocoap < eval_results['mNoCoAP']:
                self.best_mnocoap = eval_results['mNoCoAP']
            print("best eval_results['mNoCoAP']:", self.best_mnocoap)
            print_log(f"\n best eval_results['mNoCoAP']: {self.best_mnocoap}")
        else:
            raise ValueError(
                f"unsupported metric {metric}")
        return eval_results
