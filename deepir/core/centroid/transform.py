import numpy as np

def det_results_to_noco_centroids(bbox_results):
    """Transform bbox_results to centroid_results for `evaluate` in
    SIRSTDet2NoCoDataset.

    Args:
        bbox_results (list[list[np.ndarray]])

    Returns:
        centroid_results (list[np.ndarray]):
    """

    centroid_results = []
    for img_list in bbox_results:
        img_centroids = []
        for bbox in img_list:
            if bbox.shape[0] == 0:
                # list[np.ndarray]
                img_centroids.append(np.zeros((0, 3), dtype=np.float32))
            else:
                xmins = bbox[:, 0, None]
                ymins = bbox[:, 1, None]
                xmaxs = bbox[:, 2, None]
                ymaxs = bbox[:, 3, None]
                xcs = (xmins + xmaxs) / 2
                ycs = (ymins + ymaxs) / 2
                scores = bbox[:, 4, None]
                # np.ndarray
                centroid = np.concatenate((xcs, ycs, scores), axis=1)
                # list[np.ndarray]
                img_centroids.append(centroid)
        # np.ndarray
        img_centroids = np.concatenate(img_centroids, axis=0)
        # list[np.ndarray]
        centroid_results.append(img_centroids)

    return centroid_results

def det_results_to_noco_peaks(centroid_results):
    """Transform bbox_results to centroid_results for `evaluate` in
    SIRSTDet2NoCoDataset.

    Args:
        bbox_results (list[list[np.ndarray]])

    Returns:
        centroid_results (list[np.ndarray]):
    """

    centroid_results = []
    for img_list in centroid_results:
        img_centroids = []
        for centroid in img_list:
            if centroid.shape[0] == 0:
                # list[np.ndarray]
                img_centroids.append(np.zeros((0, 3), dtype=np.float32))
            else:
                cxs = centroid[:, 0, None]
                cys = centroid[:, 1, None]
                scores = centroid[:, 2, None]
                # np.ndarray
                centroid = np.concatenate((cxs, cys, scores), axis=1)
                # list[np.ndarray]
                img_centroids.append(centroid)
        # np.ndarray
        img_centroids = np.concatenate(img_centroids, axis=0)
        # list[np.ndarray]
        centroid_results.append(img_centroids)

    return centroid_results