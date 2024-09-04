from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from time import time
from .utils import _normalize_grid
from stardist.geometry.geom2d import polygons_to_label


def _ind_prob_thresh(prob, prob_thresh, b=2):
    if b is not None and np.isscalar(b):
        b = ((b,b),)*prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(slice(_bs[0] if _bs[0]>0 else None,
                         -_bs[1] if _bs[1]>0 else None)  for _bs in b)
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh


def non_maximum_suppression_inds(dist, points, scores, thresh=0.5, use_bbox=True, use_kdtree = True, verbose=1):
    """
    Applies non maximum supression to ray-convex polygons given by dists and points
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed
$ conda env create -f https://raw.githubusercontent.com/CSBDeep/CSBDeep/master/extras/environment-gpu-py3.8-tf2.4.yml
    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 2)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """

    from stardist_pkg.stardist2d import c_non_max_suppression_inds

    assert dist.ndim == 2
    assert points.ndim == 2

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    inds = c_non_max_suppression_inds(_prep(dist,  np.float32),
                                      _prep(points, np.float32),
                                      int(use_kdtree),
                                      int(use_bbox),
                                      int(verbose),
                                      np.float32(thresh))

    return inds


def non_maximum_suppression(dist, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5,
                            use_bbox=True, use_kdtree=True, verbose=False):
    """Non-Maximum-Supression of 2D polygons

    Retains only polygons whose overlap is smaller than nms_thresh

    dist.shape = (Ny,Nx, n_rays)
    prob.shape = (Ny,Nx)

    returns the retained points, probabilities, and distances:

    points, prob, dist = non_maximum_suppression(dist, prob, ....

    """

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2 and dist.ndim == 3  and prob.shape == dist.shape[:2]
    dist = np.asarray(dist)
    prob = np.asarray(prob)
    n_rays = dist.shape[-1]

    grid = _normalize_grid(grid,2)


    mask = _ind_prob_thresh(prob, prob_thresh, b)
    points = np.stack(np.where(mask), axis=1)

    dist   = dist[mask]
    scores = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    dist   = dist[ind]
    scores = scores[ind]
    points = points[ind]

    points = (points * np.array(grid).reshape((1,2)))

    if verbose:
        t = time()

    inds = non_maximum_suppression_inds(dist, points.astype(np.int32, copy=False), scores=scores,
                                        use_bbox=use_bbox, use_kdtree=use_kdtree,
                                        thresh=nms_thresh, verbose=verbose)


    _inds = inds == -1

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(inds), len(inds)))
        print("NMS took %.4f s" % (time() - t))

    return points[_inds], scores[_inds], dist[_inds], dict(suppressed=inds, points=points, scores=scores, dist=dist)