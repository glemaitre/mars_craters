from itertools import repeat

import numpy as np

from sklearn.base import BaseEstimator
from skimage.feature import blob_doh
from mahotas.features import zernike_moments

from ..iou import cc_iou


class ExtractorMixin(object):
    """Mixin class for feature extraction to modify initial X and y."""
    _estimator_type = 'extractor'

    def fit_extract(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).extract(X)
        else:
            return self.fit(X, y, **fit_params).extract(X, y)


class BlobDetector(BaseEstimator, ExtractorMixin):

    def __init__(self, min_radius=5, max_radius=40, threshold=0.01,
                 overlap=0.5, padding=1.2, confidence_level=0.5):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.threshold = threshold
        self.overlap = overlap
        self.padding = padding
        self.confidence_level = confidence_level

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

        # compute zernike moments
        return zernike_moments(patch, radius)

    def extract(self, X, y=None, **fit_params):
        candidate_blobs = blob_doh(X, min_sigma=self.min_radius,
                                   max_sigma=self.max_radius,
                                   threshold=self.threshold,
                                   overlap=self.overlap)

        # convert the candidate to list of tuple
        candidate_blobs = [tuple(blob) for blob in candidate_blobs]

        # extract feature to be returned
        features = np.array([self._extract_features(X, blob)
                             for blob in candidate_blobs])

        if y is None:
            # branch used during testing
            return features, candidate_blobs
        else:
            # find the maximum scores between each candidate and the
            # ground-truth
            scores_candidates = [max(map(cc_iou,
                                         repeat(target, len(candidate_blobs)),
                                         candidate_blobs)) for target in y]

            # threshold the scores
            labels =  [0 if score < self.confidence_level else 1
                       for score in scores_candidates]

            return features, candidate_blobs, labels
