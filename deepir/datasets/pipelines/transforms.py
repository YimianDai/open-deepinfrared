import warnings
import numpy as np
from skimage import measure as skm
import torch
import torch.nn.functional as nnf
import mmcv
from mmseg.datasets import PIPELINES as SEGPIPELINES
from mmdet.datasets import PIPELINES as DETPIPELINES
from deepir.models.builder import build_heuristic


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class RandomGammaCorrection(object):
    """Using random gamma correction to process the image.

    Args:
        gamma_lower_bound (float or int): Gamma value used in gamma correction
            lies in [gamma_lower_bound, 1/gamma_lower_bound].
            Default: 0.2.
    """

    def __init__(self, gamma_lower_bound=0.2):
        assert isinstance(gamma_lower_bound, float)
        assert gamma_lower_bound > 0 and gamma_lower_bound <= 1
        self.gamma_lower_bound = gamma_lower_bound

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        gamma = np.random.uniform(low=self.gamma_lower_bound, high=1.0)
        inv_gamma = 1.0 / gamma if np.random.rand() < 0.5 else gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')
        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class NoCoTargets(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """
    def __init__(self, mode='seg2noco', mu=0, sigma=1):
        assert isinstance(mu, float) or isinstance(mu, int)
        assert isinstance(sigma, float) or isinstance(sigma, int)
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma
        if mode in ['seg2noco', 'det2noco']:
            self.mode = mode
        else:
            raise ValueError(f"unsupported mode {mode}")

    def get_bboxes(self, gt_semantic_seg):
        bboxes = []
        gt_labels = skm.label(gt_semantic_seg, background=0)
        gt_regions = skm.regionprops(gt_labels)
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def generate_gt_noco_map(self, img, gt_bboxes):
        """Generate the ground truths for NoCo targets

        Args:
            img: np.ndarray with shape (H, W)
            gt_semantic_seg: np.ndarray with shape (H, W)
            gt_bboxes: list[list]
        """
        if len(img.shape) == 3:
            img = img.mean(-1)
        gt_noco_map = np.zeros_like(img).astype(float)
        if len(gt_bboxes) == 0:
            return gt_noco_map
        img_h, img_w = img.shape
        for _, gt_bbox in enumerate(gt_bboxes):
            # target cell coordinates
            tgt_cmin, tgt_rmin, tgt_cmax, tgt_rmax = gt_bbox
            tgt_cmin = int(tgt_cmin)
            tgt_rmin = int(tgt_rmin)
            tgt_cmax = int(tgt_cmax)
            tgt_rmax = int(tgt_rmax)
            # target cell size
            tgt_h = tgt_rmax - tgt_rmin
            tgt_w = tgt_cmax - tgt_cmin
            # Gaussian Matrix Size
            max_bg_hei = int(tgt_h * 3)
            max_bg_wid = int(tgt_w * 3)
            mesh_x, mesh_y = np.meshgrid(np.linspace(-1.5, 1.5, max_bg_wid),
                                         np.linspace(-1.5, 1.5, max_bg_hei))
            dist = np.sqrt(mesh_x*mesh_x + mesh_y*mesh_y)
            gaussian_like = np.exp(-((dist-self.mu)**2 / (2.0*self.sigma**2)))
            # unbounded background patch coordinates
            unb_bg_rmin = tgt_rmin - tgt_h
            unb_bg_rmax = tgt_rmax + tgt_h
            unb_bg_cmin = tgt_cmin - tgt_w
            unb_bg_cmax = tgt_cmax + tgt_w
            # bounded background patch coordinates
            bnd_bg_rmin = max(0, tgt_rmin - tgt_h)
            bnd_bg_rmax = min(tgt_rmax + tgt_h, img_h)
            bnd_bg_cmin = max(0, tgt_cmin - tgt_w)
            bnd_bg_cmax = min(tgt_cmax + tgt_w, img_w)
            # inds in Gaussian Matrix
            rmin_ind = bnd_bg_rmin - unb_bg_rmin
            cmin_ind = bnd_bg_cmin - unb_bg_cmin
            rmax_ind = max_bg_hei - (unb_bg_rmax - bnd_bg_rmax)
            cmax_ind = max_bg_wid - (unb_bg_cmax - bnd_bg_cmax)
            # distance weights
            bnd_gaussian = gaussian_like[rmin_ind:rmax_ind, cmin_ind:cmax_ind]

            # generate contrast weights
            tgt_cell = img[tgt_rmin:tgt_rmax, tgt_cmin:tgt_cmax]
            max_tgt = tgt_cell.max().astype(float)
            contrast_rgn = img[bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax]
            min_bg = contrast_rgn.min().astype(float)
            contrast_rgn = (contrast_rgn - min_bg) / (max_tgt - min_bg + 0.01)

            # fuse distance weights and contrast weights
            noco_rgn = bnd_gaussian * contrast_rgn
            max_noco = noco_rgn.max()
            min_noco = noco_rgn.min()
            noco_rgn = (noco_rgn - min_noco) / (max_noco - min_noco + 0.001)

            gt_noco_map[
                bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax] = noco_rgn

        return gt_noco_map

    def __call__(self, results):

        img = results['img']
        if self.mode == 'seg2noco':
            gt_semantic_seg = results['gt_semantic_seg']
            gt_bboxes = self.get_bboxes(gt_semantic_seg)
        else:
            gt_bboxes = results['gt_bboxes']
        gt_noco_map = self.generate_gt_noco_map(img, gt_bboxes)
        results['gt_noco_map'] = gt_noco_map
        if "mask_fields" in results:
            results["mask_fields"].append("gt_noco_map")

        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class Seg2DetTargets(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """
    def __init__(self, mu=0, sigma=1):
        assert isinstance(mu, float) or isinstance(mu, int)
        assert isinstance(sigma, float) or isinstance(sigma, int)
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def get_bboxes(self, gt_semantic_seg):
        bboxes = []
        gt_labels = skm.label(gt_semantic_seg, background=0)
        gt_regions = skm.regionprops(gt_labels)
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def generate_seg2det_map(self, img, gt_bboxes):
        """Generate the ground truths for NoCo targets

        Args:
            img: np.ndarray with shape (H, W)
            gt_semantic_seg: np.ndarray with shape (H, W)
            gt_bboxes: list[list]
        """
        if len(img.shape) == 3:
            img = img.mean(-1)
        gt_seg2det_map = np.zeros_like(img).astype(float)
        if len(gt_bboxes) == 0:
            return gt_seg2det_map
        img_h, img_w = img.shape
        for _, gt_bbox in enumerate(gt_bboxes):
            # target cell coordinates
            tgt_cmin, tgt_rmin, tgt_cmax, tgt_rmax = gt_bbox
            tgt_cmin = int(tgt_cmin)
            tgt_rmin = int(tgt_rmin)
            tgt_cmax = int(tgt_cmax)
            tgt_rmax = int(tgt_rmax)
            # target cell size
            tgt_h = tgt_rmax - tgt_rmin
            tgt_w = tgt_cmax - tgt_cmin
            # Gaussian Matrix Size
            max_bg_hei = int(tgt_h * 3)
            max_bg_wid = int(tgt_w * 3)
            mesh_x, mesh_y = np.meshgrid(np.linspace(-1.5, 1.5, max_bg_wid),
                                         np.linspace(-1.5, 1.5, max_bg_hei))
            dist = np.sqrt(mesh_x*mesh_x + mesh_y*mesh_y)
            gaussian_like = np.exp(-((dist-self.mu)**2 / (2.0*self.sigma**2)))
            # unbounded background patch coordinates
            unb_bg_rmin = tgt_rmin - tgt_h
            unb_bg_rmax = tgt_rmax + tgt_h
            unb_bg_cmin = tgt_cmin - tgt_w
            unb_bg_cmax = tgt_cmax + tgt_w
            # bounded background patch coordinates
            bnd_bg_rmin = max(0, tgt_rmin - tgt_h)
            bnd_bg_rmax = min(tgt_rmax + tgt_h, img_h)
            bnd_bg_cmin = max(0, tgt_cmin - tgt_w)
            bnd_bg_cmax = min(tgt_cmax + tgt_w, img_w)
            # inds in Gaussian Matrix
            rmin_ind = bnd_bg_rmin - unb_bg_rmin
            cmin_ind = bnd_bg_cmin - unb_bg_cmin
            rmax_ind = max_bg_hei - (unb_bg_rmax - bnd_bg_rmax)
            cmax_ind = max_bg_wid - (unb_bg_cmax - bnd_bg_cmax)
            # distance weights
            bnd_gaussian = gaussian_like[rmin_ind:rmax_ind, cmin_ind:cmax_ind]

            # generate contrast weights
            tgt_cell = img[tgt_rmin:tgt_rmax, tgt_cmin:tgt_cmax]
            max_tgt = tgt_cell.max().astype(float)
            contrast_rgn = img[bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax]
            min_bg = contrast_rgn.min().astype(float)
            contrast_rgn = (contrast_rgn - min_bg) / (max_tgt - min_bg + 0.01)

            # fuse distance weights and contrast weights
            noco_rgn = bnd_gaussian * contrast_rgn
            max_noco = noco_rgn.max()
            min_noco = noco_rgn.min()
            noco_rgn = (noco_rgn - min_noco) / (max_noco - min_noco + 0.001)

            ridx, cind = np.unravel_index(np.argmax(noco_rgn, axis=None),
                                          noco_rgn.shape)
            gt_seg2det_map[bnd_bg_rmin+ridx, bnd_bg_cmin+cind] = 1

        return gt_seg2det_map

    def __call__(self, results):

        img = results['img']
        gt_semantic_seg = results['gt_semantic_seg']
        gt_bboxes = self.get_bboxes(gt_semantic_seg)
        gt_semantic_seg = self.generate_seg2det_map(img, gt_bboxes)
        results['gt_semantic_seg'] = gt_semantic_seg

        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class ScaleAdaptiveGaussianTargets(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """
    def __init__(self, mode='seg2noco', mu=0, sigma=1, extend_ratio=1):
        assert isinstance(mu, float) or isinstance(mu, int)
        assert isinstance(sigma, float) or isinstance(sigma, int)
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma
        self.extend_ratio = extend_ratio
        if mode in ['seg2noco', 'det2noco']:
            self.mode = mode
        else:
            raise ValueError(f"unsupported mode {mode}")

    def get_bboxes(self, gt_semantic_seg):
        bboxes = []
        gt_labels = skm.label(gt_semantic_seg, background=0)
        gt_regions = skm.regionprops(gt_labels)
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def generate_gt_noco_map(self, img, gt_bboxes):
        """Generate the ground truths for NoCo targets

        Args:
            img: np.ndarray with shape (H, W)
            gt_semantic_seg: np.ndarray with shape (H, W)
            gt_bboxes: list[list]
        """
        if len(img.shape) == 3:
            img = img.mean(-1)
        gt_noco_map = np.zeros_like(img).astype(float)
        if len(gt_bboxes) == 0:
            return gt_noco_map
        img_h, img_w = img.shape
        for _, gt_bbox in enumerate(gt_bboxes):
            # target cell coordinates
            tgt_cmin, tgt_rmin, tgt_cmax, tgt_rmax = gt_bbox
            tgt_cmin = int(tgt_cmin)
            tgt_rmin = int(tgt_rmin)
            tgt_cmax = int(tgt_cmax)
            tgt_rmax = int(tgt_rmax)
            # target cell size
            tgt_h = tgt_rmax - tgt_rmin
            tgt_w = tgt_cmax - tgt_cmin
            # Gaussian Matrix Size
            max_bg_hei = int(tgt_h * self.extend_ratio)
            max_bg_wid = int(tgt_w * self.extend_ratio)
            mesh_x, mesh_y = np.meshgrid(np.linspace(-1.5, 1.5, max_bg_wid),
                                         np.linspace(-1.5, 1.5, max_bg_hei))
            dist = np.sqrt(mesh_x*mesh_x + mesh_y*mesh_y)
            gaussian_like = np.exp(-((dist-self.mu)**2 / (2.0*self.sigma**2)))
            # unbounded background patch coordinates
            unb_bg_rmin = tgt_rmin - tgt_h
            unb_bg_rmax = tgt_rmax + tgt_h
            unb_bg_cmin = tgt_cmin - tgt_w
            unb_bg_cmax = tgt_cmax + tgt_w
            # bounded background patch coordinates
            bnd_bg_rmin = max(0, tgt_rmin - tgt_h)
            bnd_bg_rmax = min(tgt_rmax + tgt_h, img_h)
            bnd_bg_cmin = max(0, tgt_cmin - tgt_w)
            bnd_bg_cmax = min(tgt_cmax + tgt_w, img_w)
            # inds in Gaussian Matrix
            rmin_ind = bnd_bg_rmin - unb_bg_rmin
            cmin_ind = bnd_bg_cmin - unb_bg_cmin
            rmax_ind = max_bg_hei - (unb_bg_rmax - bnd_bg_rmax)
            cmax_ind = max_bg_wid - (unb_bg_cmax - bnd_bg_cmax)
            # distance weights
            bnd_gaussian = gaussian_like[rmin_ind:rmax_ind, cmin_ind:cmax_ind]

            # fuse distance weights and contrast weights
            noco_rgn = bnd_gaussian
            max_noco = noco_rgn.max()
            min_noco = noco_rgn.min()
            noco_rgn = (noco_rgn - min_noco) / (max_noco - min_noco + 0.001)

            gt_noco_map[
                bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax] = noco_rgn

        return gt_noco_map

    def __call__(self, results):

        img = results['img']
        if self.mode == 'seg2noco':
            gt_semantic_seg = results['gt_semantic_seg']
            gt_bboxes = self.get_bboxes(gt_semantic_seg)
        else:
            gt_bboxes = results['gt_bboxes']
        gt_noco_map = self.generate_gt_noco_map(img, gt_bboxes)
        results['gt_noco_map'] = gt_noco_map
        if "mask_fields" in results:
            results["mask_fields"].append("gt_noco_map")
        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class DummySeg2NoCoTargets(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """
    def __init__(self):
        pass

    def __call__(self, results):

        gt_semantic_seg = results['gt_semantic_seg']
        results['gt_noco_map'] = gt_semantic_seg

        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class Cls2LocTargets(object):
    """Generate the ground truths for shuffle-mix without overlapping.

    Args:
        patch_size

    Example:
    >>> gt_semantic_seg = np.zeros((10, 10))
    >>> gt_semantic_seg[2:4, 2:4] = 1
    >>> results = dict(gt_semantic_seg=gt_semantic_seg)
    >>> results = Cls2LocTargets(patch_size=2)(results)
    >>> print("results['gt_cls_map']:", results['gt_cls_map'])
    """
    def __init__(self, down_sampling_rate=32):
        self.down_sampling_rate = down_sampling_rate

    def get_centroids(self, gt_semantic_seg):
        centroids = []
        gt_labels = skm.label(gt_semantic_seg, background=0)
        gt_regions = skm.regionprops(gt_labels)
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox # start from 0
            yc = (ymin + ymax) / 2
            xc = (xmin + xmax) / 2
            centroids.append([xc, yc])
        return centroids

    def __call__(self, results):
        gt_semantic_seg = results['gt_semantic_seg']
        gt_centroids = self.get_centroids(gt_semantic_seg)

        img_hei, img_wid = gt_semantic_seg.shape[:2]
        cls_hei = int(np.ceil(img_hei / self.down_sampling_rate))
        cls_wid = int(np.ceil(img_wid / self.down_sampling_rate))
        gt_cls_map = np.zeros((cls_hei, cls_wid))
        for centroid in gt_centroids:
            ccol, crow = centroid[0], centroid[1] # cx, cy
            ridx = int(np.floor(crow / self.down_sampling_rate))
            cidx = int(np.floor(ccol / self.down_sampling_rate))
            gt_cls_map[ridx, cidx] = 1
        gt_cls_map = gt_cls_map.astype(int)
        results['gt_cls_map'] = gt_cls_map
        results['down_sampling_rate'] = self.down_sampling_rate
        return results

@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class SetDownSamplingRate(object):
    """Generate the ground truths for shuffle-mix without overlapping.

    Args:
        patch_size
    """
    def __init__(self, down_sampling_rate=32):
        self.down_sampling_rate = down_sampling_rate

    def __call__(self, results):
        results['down_sampling_rate'] = self.down_sampling_rate
        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class Cls2LocShuffle(object):
    """shuffle the patches.
    Example:
    >>> gt_cls_map = np.random.randint(2, size=(5,5))
    """

    def __init__(self, patch_size=64, prob=1.0):
        self.patch_size = patch_size
        assert isinstance(prob, float)
        assert prob >= 0 and prob <= 1
        self.prob = prob

    def get_rand_inds(self, L, ratio):
        assert isinstance(L, int)
        assert L > 0

        orig_inds = torch.arange(L)
        logic_inds = np.random.random_sample(L) < ratio
        select_inds = orig_inds[logic_inds] # inds to be shuffled
        select_len = select_inds.shape[-1]
        shuffled_inds = orig_inds.clone()
        shuffled_inds[logic_inds] = select_inds[torch.randperm(select_len)]

        return shuffled_inds

    def __call__(self, results):
        # Step 1：先将 img 和 gt_cls_map 转化成 4D Tensor
        # img.shape: (800, 800), uint8
        img = results['img'][None, None, ...].astype(float)
        img = torch.from_numpy(img)
        # gt_cls_map.shape: (25, 25), int64
        gt_cls_map = results['gt_cls_map'][None, None, ...].astype(float)
        gt_cls_map = torch.from_numpy(gt_cls_map)
        patch_size = self.patch_size
        assert patch_size % results['down_sampling_rate'] == 0
        cls_kernel_size = int(patch_size / results['down_sampling_rate'])
        # Step 2：unfold
        img_u = nnf.unfold(img, kernel_size=patch_size, stride=patch_size,
                           padding=0) # (B, ps*ps, L)
        cls_u = nnf.unfold(gt_cls_map, kernel_size=cls_kernel_size,
                           stride=cls_kernel_size, padding=0)
        L = img_u.shape[-1]
        assert img_u.shape[-1] == cls_u.shape[-1]

        # Step 3: shuffle patches in a certain ratio
        p_img_u, p_cls_u = [], []
        for b in range(img.shape[0]):
            rand_inds = self.get_rand_inds(L, self.prob)
            p_img_u.append(img_u[b][:, rand_inds][None, ...])
            p_cls_u.append(cls_u[b][:, rand_inds][None, ...])
        p_img_u = torch.cat(p_img_u, dim=0)
        p_cls_u = torch.cat(p_cls_u, dim=0)

        # Step 4：fold the patches into images
        img_f = nnf.fold(p_img_u, img.shape[-2:], kernel_size=patch_size,
                         stride=patch_size, padding=0)
        cls_f = nnf.fold(p_cls_u, gt_cls_map.shape[-2:],
                         kernel_size=cls_kernel_size, stride=cls_kernel_size,
                         padding=0)

        # print("img == img_f:\n", img == img_f)
        # print("gt_cls_map == cls_f:\n", gt_cls_map == cls_f)
        # Step 5：最后将 img 和 gt_cls_map 从 4D Tensor 转化成 2D np.ndarray
        img_f = np.squeeze(img_f.cpu().numpy(), axis=(0,1)).astype(np.uint8)
        cls_f = np.squeeze(cls_f.cpu().numpy(), axis=(0,1)).astype(int)
        results['img'] = img_f
        results['gt_cls_map'] = cls_f

        return results

@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class ShuffleMix(object):
    """shuffle the patches.
    Example:
    >>> gt_cls_map = np.random.randint(2, size=(5,5))
    """

    def __init__(self, patch_size=64, orig_shuffle_ratio=0.0,
                 copy_paste_prob=0.25,
                 alpha=0.2, weight_mode='beta', ):

        assert isinstance(patch_size, int)
        assert isinstance(copy_paste_prob, float)
        assert copy_paste_prob >= 0 and copy_paste_prob <= 1
        assert isinstance(alpha, float) and alpha > 0
        self.orig_shuffle_ratio = orig_shuffle_ratio
        self.patch_size = patch_size
        self.copy_paste_prob = copy_paste_prob
        self.alpha = alpha
        assert weight_mode in ['beta', 'copy-paste']
        self.weight_mode = weight_mode

    def get_rand_inds(self, L, ratio):
        assert isinstance(L, int)
        assert L > 0

        orig_inds = torch.arange(L)
        logic_inds = np.random.random_sample(L) < ratio
        select_inds = orig_inds[logic_inds] # inds to be shuffled
        select_len = select_inds.shape[-1]
        shuffled_inds = orig_inds.clone()
        shuffled_inds[logic_inds] = select_inds[torch.randperm(select_len)]

        return shuffled_inds

    def get_target_masked_rand_inds(self, target_inds, L, target_mix_prob):

        target_num = len(target_inds) # 图像中的目标个数
        # 用于 shuffle 的 inds，部分元素 将会被替换成目标 patch 的 inds
        # shuffle_inds = torch.randperm(L)
        shuffle_inds = self.get_rand_inds(L, self.orig_shuffle_ratio)

        # 挑选出哪些 patch 要被 target patch 所替换
        tmp_rand_inds = torch.randperm(L) # 用于采样填充目标的 inds
        target_paste_times = int(L * target_mix_prob)
        select_inds = tmp_rand_inds[:target_paste_times]
        shuffle_target_inds = shuffle_inds[select_inds]

        # 将目标 patch 的 inds 替换进 shuffle_inds
        for i in range(target_num):
            shuffle_target_inds[i::target_num] = target_inds[i]
        shuffle_inds[select_inds] = shuffle_target_inds

        return shuffle_inds

    def __call__(self, results):
        # Step 1：先将 img 和 gt_cls_map 转化成 4D Tensor
        # img.shape: (800, 800), uint8
        img = results['img'][None, None, ...].astype(float)
        img = torch.from_numpy(img)
        # gt_cls_map.shape: (25, 25), int64
        gt_cls_map = results['gt_cls_map'][None, None, ...].astype(float)
        gt_cls_map = torch.from_numpy(gt_cls_map)
        patch_size = self.patch_size
        assert patch_size % results['down_sampling_rate'] == 0
        cls_kernel_size = int(patch_size / results['down_sampling_rate'])

        # Step 2：unfold
        img_u = nnf.unfold(img, kernel_size=patch_size, stride=patch_size,
                           padding=0) # (B, ps*ps, L)
        cls_u = nnf.unfold(gt_cls_map, kernel_size=cls_kernel_size,
                           stride=cls_kernel_size, padding=0)
        L = img_u.shape[-1]
        assert img_u.shape[-1] == cls_u.shape[-1]

        # Step 3: shuffle patches in a certain ratio
        p_img_u, p_cls_u = [], []
        for b in range(img.shape[0]):
            # rand_inds = self.get_rand_inds(L, self.prob)
            target_inds = (cls_u[b] == 1).nonzero()[:, -1].unique()
            rand_inds = self.get_target_masked_rand_inds(target_inds, L,
                                                         self.copy_paste_prob)
            p_img_u.append(img_u[b][:, rand_inds][None, ...])
            p_cls_u.append(cls_u[b][:, rand_inds][None, ...])
        p_img_u = torch.cat(p_img_u, dim=0)
        p_cls_u = torch.cat(p_cls_u, dim=0)

        # Step 4: mixup images
        if self.weight_mode == 'beta':
            lam = torch.from_numpy(np.random.beta(self.alpha, self.alpha, L))
        elif self.weight_mode == 'copy-paste':
            lam = 0
        else:
            raise ValueError('Unknown weight_mode')
        mixed_img_u = lam * img_u + (1 - lam) * p_img_u
        # mixed_img_u = (img_u**(lam)) * (p_img_u**(1 - lam))
        mixed_cls_u = lam * cls_u + (1 - lam) * p_cls_u

        # Step 4：fold the patches into images
        img_f = nnf.fold(mixed_img_u, img.shape[-2:], kernel_size=patch_size,
                         stride=patch_size, padding=0)
        cls_f = nnf.fold(mixed_cls_u, gt_cls_map.shape[-2:],
                         kernel_size=cls_kernel_size, stride=cls_kernel_size,
                         padding=0)

        # Step 5：最后将 img 和 gt_cls_map 从 4D Tensor 转化成 2D np.ndarray
        img_f = np.squeeze(img_f.cpu().numpy(), axis=(0,1))
        cls_f = np.squeeze(cls_f.cpu().numpy(), axis=(0,1))
        results['img'] = img_f
        results['gt_cls_map'] = cls_f
        results['patch_size'] = patch_size

        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class InnerPatchShuffle(object):
    """shuffle the patches.
    """
    def __init__(self, patch_size=None):
        self.patch_size = patch_size

    def shuffle_patch(self, img_patch, cls_patch, cell_size):
        # Step 2：unfold
        img_u = nnf.unfold(img_patch, kernel_size=cell_size, stride=cell_size,
                           padding=0) # (B, ps*ps, L)
        cls_u = nnf.unfold(cls_patch, kernel_size=1, stride=1, padding=0)
        L = img_u.shape[-1]
        assert img_u.shape[-1] == cls_u.shape[-1]

        # Step 3: shuffle patches in a certain ratio
        p_img_u, p_cls_u = [], []
        for b in range(img_patch.shape[0]):
            rand_inds = torch.randperm(L)
            p_img_u.append(img_u[b][:, rand_inds][None, ...])
            p_cls_u.append(cls_u[b][:, rand_inds][None, ...])
        p_img_u = torch.cat(p_img_u, dim=0)
        p_cls_u = torch.cat(p_cls_u, dim=0)

        # Step 4：fold the patches into images
        img_f = nnf.fold(p_img_u, img_patch.shape[-2:], kernel_size=cell_size,
                         stride=cell_size, padding=0)
        cls_f = nnf.fold(p_cls_u, cls_patch.shape[-2:], kernel_size=1,
                         stride=1, padding=0)

        return img_f, cls_f

    def __call__(self, results):
        # img.shape: (800, 800), uint8
        img = results['img'][None, None, ...].astype(float)
        img = torch.from_numpy(img)
        # gt_cls_map.shape: (25, 25), int64
        gt_cls_map = results['gt_cls_map'][None, None, ...].astype(float)
        gt_cls_map = torch.from_numpy(gt_cls_map)
        down_sampling_rate = results['down_sampling_rate']
        if self.patch_size is None:
            patch_size = results['patch_size']
        else:
            patch_size = self.patch_size
        # print('patch_size:', patch_size)
        # print('down_sampling_rate:', down_sampling_rate)
        assert patch_size % down_sampling_rate == 0
        cell_num = int(patch_size / down_sampling_rate)
        cell_size = down_sampling_rate

        rc_inds = gt_cls_map.nonzero()[:,2:4]
        for r, c in rc_inds:
            cls_r_beg = int((r / cell_size).floor())
            cls_c_beg = int((c / cell_size).floor())
            cls_r_end = cls_r_beg + cell_num
            cls_c_end = cls_c_beg + cell_num
            cls_patch = gt_cls_map[:, :, cls_r_beg:cls_r_end,
                                         cls_c_beg:cls_c_end]
            img_r_beg = cls_r_beg * cell_size
            img_c_beg = cls_c_beg * cell_size
            img_r_end = cls_r_end * cell_size
            img_c_end = cls_c_end * cell_size
            img_patch = img[:, :, img_r_beg:img_r_end, img_c_beg:img_c_end]
            shuffled_img_patch, shuffled_cls_patch = self.shuffle_patch(
                img_patch, cls_patch, cell_size)
            img[:, :,
                img_r_beg:img_r_end, img_c_beg:img_c_end] = shuffled_img_patch
            gt_cls_map[:, :,
                cls_r_beg:cls_r_end, cls_c_beg:cls_c_end] = shuffled_cls_patch

        # 最后将 img 和 gt_cls_map 从 4D Tensor 转化成 2D np.ndarray
        img = np.squeeze(img.cpu().numpy(), axis=(0,1))
        gt_cls_map = np.squeeze(gt_cls_map.cpu().numpy(), axis=(0,1))
        results['img'] = img
        results['gt_cls_map'] = gt_cls_map
        return results


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class PopMix(object):
    def __init__(self, patch_size=32,
                 tgt_copy_paste_prob=0.0625,
                 bg_copy_paste_prob=0.0625,
                 orig_shuffle_ratio=0.,
                 alpha=0.2,
                 weight_mode='beta',
                 heuristic=dict(type='TopHat', footprint='disk', radius=2)):
        self.patch_size = patch_size
        self.alpha = alpha
        self.tgt_copy_paste_prob = tgt_copy_paste_prob
        self.bg_copy_paste_prob = bg_copy_paste_prob
        self.orig_shuffle_ratio = orig_shuffle_ratio
        self.heuristic = build_heuristic(heuristic)
        assert weight_mode in ['beta', 'copy-paste']
        self.weight_mode = weight_mode

    def __call__(self, results):

        # Step 1：先将 img 和 gt_cls_map 转化成 4D Tensor
        # img.shape: (800, 800), uint8
        img = results['img'][None, None, ...].astype(float)
        img = torch.from_numpy(img)
        # gt_cls_map.shape: (25, 25), int64
        gt_cls_map = results['gt_cls_map'][None, None, ...].astype(float)
        gt_cls_map = torch.from_numpy(gt_cls_map)
        # ori_img.shape: (320, 256), uint8
        resized_shape = results['scale']
        ori_img = results['ori_img']
        peak_img = self.heuristic(ori_img)
        saliency_map, _, _ = mmcv.imresize(peak_img, resized_shape,
                                           return_scale=True)
        saliency_map = saliency_map[None, None, ...].astype(float)
        saliency_map = torch.from_numpy(saliency_map)
        patch_size = self.patch_size
        assert patch_size % results['down_sampling_rate'] == 0
        cls_kernel_size = int(patch_size / results['down_sampling_rate'])

        # Step 2：unfold
        img_u = nnf.unfold(img, kernel_size=patch_size,
                           stride=patch_size,
                           padding=0) # (B, ps*ps, L)
        cls_u = nnf.unfold(gt_cls_map, kernel_size=1, stride=1, padding=0)
        sal_u = nnf.unfold(saliency_map, kernel_size=patch_size,
                           stride=patch_size,
                           padding=0) # (B, ps*ps, L)
        L = img_u.shape[-1]
        assert img_u.shape[-1] == cls_u.shape[-1]

        p_img_u, p_cls_u = [], []
        for b in range(img.shape[0]):
            target_paste_times = int(L * self.tgt_copy_paste_prob)
            bg_paste_times = int(L * self.bg_copy_paste_prob)
            target_inds = (cls_u[b] == 1).nonzero()[:, -1].unique()
            # set the target saliency to zero
            sal_u[b][:, target_inds] = 0
            sal_prob = sal_u[b].sum(axis=0) / sal_u[b].sum()
            sal_prob = sal_prob.cpu().numpy()
            bg_inds = np.random.choice(L, bg_paste_times, p=sal_prob)
            bg_inds = torch.from_numpy(bg_inds)
            rand_inds = self.get_masked_inds(target_inds, bg_inds, L,
                                             target_paste_times, bg_paste_times)
            p_img_u.append(img_u[b][:, rand_inds][None, ...])
            p_cls_u.append(cls_u[b][:, rand_inds][None, ...])
        p_img_u = torch.cat(p_img_u, dim=0)
        p_cls_u = torch.cat(p_cls_u, dim=0)

        # Step 4: mixup images
        if self.weight_mode == 'beta':
            lam = torch.from_numpy(np.random.beta(self.alpha, self.alpha, L))
        elif self.weight_mode == 'copy-paste':
            lam = 0
        else:
            raise ValueError('Unknown weight_mode')
        mixed_img_u = lam * img_u + (1 - lam) * p_img_u
        mixed_cls_u = lam * cls_u + (1 - lam) * p_cls_u
        # mixed_cls_u = p_cls_u

        # Step 4：fold the patches into images
        img_f = nnf.fold(mixed_img_u, img.shape[-2:],
                         kernel_size=self.patch_size,
                         stride=self.patch_size, padding=0)
        cls_f = nnf.fold(mixed_cls_u, gt_cls_map.shape[-2:],
                         kernel_size=cls_kernel_size, stride=cls_kernel_size,
                         padding=0)

        # Step 5：最后将 img 和 gt_cls_map 从 4D Tensor 转化成 2D np.ndarray
        img_f = np.squeeze(img_f.cpu().numpy(), axis=(0,1))
        cls_f = np.squeeze(cls_f.cpu().numpy(), axis=(0,1))
        results['img'] = img_f
        results['gt_cls_map'] = cls_f
        results['patch_size'] = patch_size

        return results

    def get_masked_inds(self, target_inds, bg_inds, L,
                        target_paste_times, bg_paste_times):

        target_num = len(target_inds) # 图像中的目标个数
        # 用于 shuffle 的 inds，部分元素 将会被替换成目标 patch 的 inds
        # shuffle_inds = torch.randperm(L)
        shuffle_inds = self.get_rand_inds(L, self.orig_shuffle_ratio)

        # 挑选出哪些 patch 要被 target patch 所替换
        tmp_rand_inds = torch.randperm(L) # 用于采样填充目标的 inds
        select_inds = tmp_rand_inds[:target_paste_times+bg_paste_times]
        shuffle_target_inds = shuffle_inds[select_inds[:target_paste_times]]

        # 将目标 patch 的 inds 替换进 shuffle_inds
        for i in range(target_num):
            shuffle_target_inds[i::target_num] = target_inds[i]
        shuffle_inds[select_inds[:target_paste_times]] = shuffle_target_inds
        shuffle_inds[select_inds[target_paste_times:]] = bg_inds
        return shuffle_inds

    def get_rand_inds(self, L, ratio):
        assert isinstance(L, int)
        assert L > 0

        orig_inds = torch.arange(L)
        logic_inds = np.random.random_sample(L) < ratio
        select_inds = orig_inds[logic_inds] # inds to be shuffled
        select_len = select_inds.shape[-1]
        shuffled_inds = orig_inds.clone()
        shuffled_inds[logic_inds] = select_inds[torch.randperm(select_len)]

        return shuffled_inds


@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class MultMix(object):
    def __init__(self, patch_size=32,
                 tgt_copy_paste_prob=0.0625,
                 orig_shuffle_ratio=0.):
        self.patch_size = patch_size
        self.tgt_copy_paste_prob = tgt_copy_paste_prob
        self.orig_shuffle_ratio = orig_shuffle_ratio

    def __call__(self, results):

        if results['gt_cls_map'].sum() == 0:
            results['patch_size'] = self.patch_size
            img = results['img'][None, None, ...].astype(float)
            img = torch.from_numpy(img)
            img_f = np.squeeze(img.cpu().numpy(), axis=(0,1))
            gt_cls_map = results['gt_cls_map'][None, None, ...].astype(float)
            gt_cls_map = torch.from_numpy(gt_cls_map)
            cls_f = np.squeeze(gt_cls_map.cpu().numpy(), axis=(0,1))
            results['img'] = img_f
            results['gt_cls_map'] = cls_f
            return results
        else:
            # Step 1：先将 img 和 gt_cls_map 转化成 4D Tensor
            # img.shape: (800, 800), uint8
            img = results['img'][None, None, ...].astype(float)
            img = torch.from_numpy(img)
            # gt_cls_map.shape: (25, 25), int64
            gt_cls_map = results['gt_cls_map'][None, None, ...].astype(float)
            gt_cls_map = torch.from_numpy(gt_cls_map)
            patch_size = self.patch_size
            assert patch_size % results['down_sampling_rate'] == 0
            cls_kernel_size = int(patch_size / results['down_sampling_rate'])

            # Step 2：unfold
            img_u = nnf.unfold(img, kernel_size=patch_size,
                            stride=patch_size,
                            padding=0) # (B, ps*ps, L)
            cls_u = nnf.unfold(gt_cls_map, kernel_size=1, stride=1, padding=0)
            L = img_u.shape[-1]
            assert img_u.shape[-1] == cls_u.shape[-1]

            m_img_u, m_cls_u = [], []
            for b in range(img.shape[0]):
                target_paste_times = int(L * self.tgt_copy_paste_prob)
                target_inds = (cls_u[b] == 1).nonzero()[:, -1].unique()
                target_num = len(target_inds)
                select_bg_inds = torch.randperm(L)[:target_paste_times]
                select_fg_inds = select_bg_inds.clone()
                # 将目标 patch 的 inds 替换进 shuffle_inds
                for i in range(target_num):
                    select_fg_inds[i::target_num] = target_inds[i]
                fg_u = img_u[b][:, select_fg_inds]
                bg_u = img_u[b][:, select_bg_inds]
                bg_cls_u = cls_u[b][:, select_bg_inds]
                fg_cls_u = cls_u[b][:, select_fg_inds]
                lam1 = torch.from_numpy(np.random.beta(0.2, 0.2, target_paste_times))
                # mixed_img_u = lam * fg_u + (1 - lam) * bg_u
                mixed_fg_u_1 = fg_u - fg_u.mean(axis=0) + bg_u
                mixed_fg_u_2 = fg_u / (fg_u.mean(axis=0)+1) * bg_u
                lam2 = torch.rand(target_paste_times)
                mixed_fg_u = lam2 * mixed_fg_u_1 + (1-lam2) * mixed_fg_u_2
                mixed_img_u = lam1 * mixed_fg_u + (1-lam1) * bg_u
                mixed_cls_u = lam1 * fg_cls_u + (1 - lam1) * bg_cls_u
                # mixed_cls_u = lam * fg_cls_u
                # fg_dist = fg_u / (fg_u.mean(axis=0) + 0.001)
                # ratio  = bg_u.mean(axis=0) / (fg_u.mean(axis=0) + 0.001)
                # ratio = ratio / (ratio.max() + 0.1)
                # mix_bg_u = bg_u * fg_dist
                # print('mix_bg_u:', mix_bg_u)
                img_u[b][:, select_bg_inds] = mixed_img_u
                cls_u[b][:, select_bg_inds] = mixed_cls_u
                m_img_u.append(img_u[b][None, ...])
                m_cls_u.append(cls_u[b][None, ...])
            m_img_u = torch.cat(m_img_u, dim=0)
            m_cls_u = torch.cat(m_cls_u, dim=0)

            # Step 4：fold the patches into images
            img_f = nnf.fold(m_img_u, img.shape[-2:],
                            kernel_size=self.patch_size,
                            stride=self.patch_size, padding=0)
            cls_f = nnf.fold(m_cls_u, gt_cls_map.shape[-2:],
                            kernel_size=cls_kernel_size, stride=cls_kernel_size,
                            padding=0)

            # Step 5：最后将 img 和 gt_cls_map 从 4D Tensor 转化成 2D np.ndarray
            img_f = np.squeeze(img_f.cpu().numpy(), axis=(0,1))
            cls_f = np.squeeze(cls_f.cpu().numpy(), axis=(0,1))
            results['img'] = img_f
            results['gt_cls_map'] = cls_f
            results['patch_size'] = patch_size

            return results

@SEGPIPELINES.register_module()
@DETPIPELINES.register_module()
class NoCoPeaks(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """
    def __init__(self, mode='det2noco', mu=0, sigma=1):
        assert isinstance(mu, float) or isinstance(mu, int)
        assert isinstance(sigma, float) or isinstance(sigma, int)
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma
        if mode in ['seg2noco', 'det2noco']:
            self.mode = mode
        else:
            raise ValueError(f"unsupported mode {mode}")

    def generate_gt_noco_peak(self, img, gt_bboxes):
        """Generate the ground truths for NoCo targets

        Args:
            img: np.ndarray with shape (H, W)
            gt_semantic_seg: np.ndarray with shape (H, W)
            gt_bboxes: list[list]
        """
        gt_noco_peak_coords = []
        if len(img.shape) == 3:
            img = (img[..., 0] + img[..., 1] + img[..., 2]) / 3
        gt_noco_map = np.zeros_like(img).astype(float)
        if len(gt_bboxes) == 0:
            return gt_noco_map
        img_h, img_w = img.shape
        for _, gt_bbox in enumerate(gt_bboxes):
            # target cell coordinates
            tgt_cmin, tgt_rmin, tgt_cmax, tgt_rmax = gt_bbox
            tgt_cmin = int(tgt_cmin)
            tgt_rmin = int(tgt_rmin)
            tgt_cmax = int(tgt_cmax)
            tgt_rmax = int(tgt_rmax)
            # target cell size
            tgt_h = tgt_rmax - tgt_rmin
            tgt_w = tgt_cmax - tgt_cmin
            # Gaussian Matrix Size
            max_bg_hei = int(tgt_h * 3)
            max_bg_wid = int(tgt_w * 3)
            mesh_x, mesh_y = np.meshgrid(np.linspace(-1.5, 1.5, max_bg_wid),
                                         np.linspace(-1.5, 1.5, max_bg_hei))
            dist = np.sqrt(mesh_x*mesh_x + mesh_y*mesh_y)
            gaussian_like = np.exp(-((dist-self.mu)**2 / (2.0*self.sigma**2)))
            # unbounded background patch coordinates
            unb_bg_rmin = tgt_rmin - tgt_h
            unb_bg_rmax = tgt_rmax + tgt_h
            unb_bg_cmin = tgt_cmin - tgt_w
            unb_bg_cmax = tgt_cmax + tgt_w
            # bounded background patch coordinates
            bnd_bg_rmin = max(0, tgt_rmin - tgt_h)
            bnd_bg_rmax = min(tgt_rmax + tgt_h, img_h)
            bnd_bg_cmin = max(0, tgt_cmin - tgt_w)
            bnd_bg_cmax = min(tgt_cmax + tgt_w, img_w)
            # inds in Gaussian Matrix
            rmin_ind = bnd_bg_rmin - unb_bg_rmin
            cmin_ind = bnd_bg_cmin - unb_bg_cmin
            rmax_ind = max_bg_hei - (unb_bg_rmax - bnd_bg_rmax)
            cmax_ind = max_bg_wid - (unb_bg_cmax - bnd_bg_cmax)
            # distance weights
            bnd_gaussian = gaussian_like[rmin_ind:rmax_ind, cmin_ind:cmax_ind]

            # generate contrast weights
            tgt_cell = img[tgt_rmin:tgt_rmax, tgt_cmin:tgt_cmax]
            max_tgt = tgt_cell.max().astype(float)
            contrast_rgn = img[bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax]
            min_bg = contrast_rgn.min().astype(float)
            contrast_rgn = (contrast_rgn - min_bg) / (max_tgt - min_bg + 0.01)

            # fuse distance weights and contrast weights
            noco_rgn = bnd_gaussian * contrast_rgn
            max_noco = noco_rgn.max()
            min_noco = noco_rgn.min()
            noco_rgn = (noco_rgn - min_noco) / (max_noco - min_noco + 0.001)

            ridx, cind = np.unravel_index(np.argmax(noco_rgn, axis=None),
                                          noco_rgn.shape)
            # output: (x, y)
            gt_noco_peak_coords.append([bnd_bg_cmin+cind, bnd_bg_rmin+ridx])

        return gt_noco_peak_coords

    def __call__(self, results):

        img = results['img']
        gt_bboxes = results['gt_bboxes']
        gt_noco_peak = self.generate_gt_noco_peak(img, gt_bboxes)
        results['gt_kpts'] = gt_noco_peak

        return results


@DETPIPELINES.register_module()
class OSCARPad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            # results[key] = results[key].pad(pad_shape, pad_val=pad_val)
            results[key] = mmcv.impad(
                    results[key], shape=pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str