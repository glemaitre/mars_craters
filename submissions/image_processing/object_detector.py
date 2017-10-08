from __future__ import division

import math
from itertools import repeat

import numpy as np

from joblib import Parallel, delayed

from sklearn.base import clone, BaseEstimator
from imblearn.ensemble import BalancedBaggingClassifier

from skimage.feature import blob_doh
from mahotas.features import zernike_moments
from mahotas.features import surf

###############################################################################
# IOU function


def cc_iou(circle1, circle2):
    """
    Intersection over Union (IoU) between two circles

    Parameters
    ----------
    circle1 : tuple of floats
        first circle parameters (x_pos, y_pos, radius)
    circle2 : tuple of floats
        second circle parameters (x_pos, y_pos, radius)

    Returns
    -------
    float
        ratio between area of intersection and area of union

    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    d = math.hypot(x2 - x1, y2 - y1)

    area_intersection = cc_intersection(d, r1, r2)
    area_union = math.pi * (r1 * r1 + r2 * r2) - area_intersection

    return area_intersection / area_union


def cc_intersection(dist, rad1, rad2):
    """
    Area of intersection between two circles

    Parameters
    ----------
    dist : positive float
        distance between circle centers
    rad1 : positive float
        radius of first circle
    rad2 : positive float
        radius of second circle

    Returns
    -------
    intersection_area : positive float
        area of intersection between circles

    References
    ----------
    http://mathworld.wolfram.com/Circle-CircleIntersection.html

    """
    if dist < 0:
        raise ValueError("Distance between circles must be positive")
    if rad1 < 0 or rad2 < 0:
        raise ValueError("Circle radius must be positive")

    if dist == 0 or (dist <= abs(rad2 - rad1)):
        return min(rad1, rad2) ** 2 * math.pi

    if dist > rad1 + rad2 or rad1 == 0 or rad2 == 0:
        return 0

    rad1_sq = rad1 * rad1
    rad2_sq = rad2 * rad2

    circle1 = rad1_sq * math.acos((dist * dist + rad1_sq - rad2_sq) /
                                  (2 * dist * rad1))
    circle2 = rad2_sq * math.acos((dist * dist + rad2_sq - rad1_sq) /
                                  (2 * dist * rad2))
    intersec = 0.5 * math.sqrt((-dist + rad1 + rad2) * (dist + rad1 - rad2) *
                               (dist - rad1 + rad2) * (dist + rad1 + rad2))
    intersection_area = circle1 + circle2 + intersec

    return intersection_area

###############################################################################
# Extractor


class ExtractorMixin(object):
    """Mixin class for feature extraction to modify initial X and y."""
    _estimator_type = 'extractor'

    def fit_extract(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).extract(X)
        else:
            return self.fit(X, y, **fit_params).extract(X, y)


class BlobExtractor(BaseEstimator, ExtractorMixin):

    def __init__(self, min_radius=5, max_radius=40, blob_threshold=0.01,
                 overlap=0.5, padding=1.2, iou_threshold=0.5):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.blob_threshold = blob_threshold
        self.overlap = overlap
        self.padding = padding
        self.iou_threshold = iou_threshold

    def fit(self, X, y=None, **fit_params):
        # This extractor does not require any fitting
        return self

    def _extract_features(self, X, candidate):

        y, x, radius = int(candidate[0]), int(candidate[1]), candidate[2]
        padded_radius = int(self.padding * radius)

        # compute the coordinate of the patch to select
        x_min = x - padded_radius
        x_min = x_min if x_min < 0 else 0
        y_min = y - padded_radius
        y_min = y_min if y_min < 0 else 0
        x_max = x + padded_radius
        x_max = x_max if x_max > X.shape[0] else X.shape[0] - 1
        y_max = y + padded_radius
        y_max = y_max if y_max > X.shape[1] else X.shape[1] - 1

        patch = X[y_min:y_max, x_min:x_max]

        # compute Zernike moments
        zernike = zernike_moments(patch, radius)

        # compute SURF descriptor
        keypoint = np.array([[y, x, 1, 0.1, 1]])
        surf_descriptor = surf.descriptors(patch, keypoint,
                                           is_integral=False).ravel()
        if not surf_descriptor.size:
            surf_descriptor = np.zeros((70, ))

        return np.hstack((zernike, surf_descriptor))

    def extract(self, X, y=None, **fit_params):
        candidate_blobs = blob_doh(X, min_sigma=self.min_radius,
                                   max_sigma=self.max_radius,
                                   threshold=self.blob_threshold,
                                   overlap=self.overlap)

        # convert the candidate to list of tuple
        candidate_blobs = [tuple(blob) for blob in candidate_blobs]

        # extract feature to be returned
        features = [self._extract_features(X, blob)
                    for blob in candidate_blobs]

        if y is None:
            # branch used during testing
            return features, candidate_blobs, [None] * len(features)
        elif not y:
            # branch if there is no crater in the image
            labels = [0] * len(candidate_blobs)

            return features, candidate_blobs, labels
        else:
            # case the we did not detect any blobs
            if not len(features):
                return ([], [], [])

            # find the maximum scores between each candidate and the
            # ground-truth
            scores_candidates = [max(map(cc_iou, repeat(blob, len(y)), y))
                                 for blob in candidate_blobs]

            # threshold the scores
            labels = [0 if score < self.iou_threshold else 1
                      for score in scores_candidates]

            return features, candidate_blobs, labels

###############################################################################
# Detector


class ObjectDetector(object):
    def __init__(self, extractor=None, estimator=None, n_jobs=1):
        self.extractor = extractor
        self.estimator = estimator
        self.n_jobs = n_jobs

    def _extract_features(self, X, y):
        # extract feature for all the image containing craters
        data_extracted = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extractor_.fit_extract)(image, craters)
            for image, craters in zip(X, y))

        # organize the data to fit it inside the classifier
        data, location, target, idx_cand_to_img = [], [], [], []
        for img_idx, candidate in enumerate(data_extracted):
            # check if this is an empty features
            if len(candidate[0]):
                data.append(np.vstack(candidate[0]))
                location += candidate[1]
                target += candidate[2]
                idx_cand_to_img += [img_idx] * len(candidate[1])
        # convert to numpy array the data needed to feed the classifier
        data = np.concatenate(data)
        target = np.array(target)

        return data, location, target, idx_cand_to_img

    def fit(self, X, y):
        if self.extractor is None:
            self.extractor_ = BlobExtractor()
        else:
            self.extractor_ = clone(self.extractor)

        if self.estimator is None:
            self.estimator_ = BalancedBaggingClassifier(n_jobs=self.n_jobs)
        else:
            self.estimator_ = clone(self.estimator)

        # extract the features for the training data
        data, _, target, _ = self._extract_features(X, y)

        # fit the underlying classifier
        self.estimator_.fit(data, target)

        return self

    def predict(self, X):
        # extract the data for the current image
        data, location, _, idx_cand_to_img = self._extract_features(
            X, [None] * len(X))

        # classify each candidate
        y_pred = self.estimator_.predict_proba(data)

        # organize the output
        output = [[] for _ in range(len(X))]
        crater_idx = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        for crater, pred, img_idx in zip(location, y_pred, idx_cand_to_img):
            output[img_idx].append((crater[0], crater[1], crater[2],
                                    pred[crater_idx]))

        return output
