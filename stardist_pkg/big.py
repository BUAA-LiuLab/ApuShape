import numpy as np
import warnings
import math
from tqdm import tqdm
from skimage.measure import regionprops
from skimage.draw import polygon
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict
from itertools import product




OBJECT_KEYS = set(('prob', 'points', 'coord', 'dist', 'class_prob', 'class_id'))
COORD_KEYS = set(('points', 'coord'))



class Block:
    """One-dimensional block as part of a chain.

    There are no explicit start and end positions. Instead, each block is
    aware of its predecessor and successor and derives such things (recursively)
    based on its neighbors.

    Blocks overlap with one another (at least min_overlap + 2*context) and
    have a read region (the entire block) and a write region (ignoring context).
    Given a query interval, Block.is_responsible will return true for only one
    block of a chain (or raise an exception if the interval is larger than
    min_overlap or even the entire block without context).

    """
    def __init__(self, size, min_overlap, context, pred):
        self.size = int(size)
        self.min_overlap = int(min_overlap)
        self.context = int(context)
        self.pred = pred
        self.succ = None
        assert 0 <= self.min_overlap + 2*self.context < self.size
        self.stride = self.size - (self.min_overlap + 2*self.context)
        self._start = 0
        self._frozen = False

    @property
    def start(self):
        return self._start if (self.frozen or self.at_begin) else self.pred.succ_start

    @property
    def end(self):
        return self.start + self.size

    @property
    def succ_start(self):
        return self.start + self.stride

    def add_succ(self):
        assert self.succ is None and not self.frozen
        self.succ = Block(self.size, self.min_overlap, self.context, self)
        return self.succ

    def decrease_stride(self, amount):
        amount = int(amount)
        assert 0 <= amount < self.stride and not self.frozen
        self.stride -= amount

    def freeze(self):
        """Call on first block to freeze entire chain (after construction is done)"""
        assert not self.frozen and (self.at_begin or self.pred.frozen)
        self._start = self.start
        self._frozen = True
        if not self.at_end:
            self.succ.freeze()

    @property
    def slice_read(self):
        return slice(self.start, self.end)

    @property
    def slice_crop_context(self):
        """Crop context relative to read region"""
        return slice(self.context_start, self.size - self.context_end)

    @property
    def slice_write(self):
        return slice(self.start + self.context_start, self.end - self.context_end)

    def is_responsible(self, bbox):
        """Responsibility for query interval bbox, which is assumed to be smaller than min_overlap.

        If the assumption is met, only one block of a chain will return true.
        If violated, one or more blocks of a chain may raise a NotFullyVisible exception.
        The exception will have an argument that is
            False if bbox is larger than min_overlap, and
            True if bbox is even larger than the entire block without context.

        bbox: (int,int)
            1D bounding box interval with coordinates relative to size without context

        """
        bmin, bmax = bbox

        r_start = 0 if self.at_begin else (self.pred.overlap - self.pred.context_end - self.context_start)
        r_end = self.size - self.context_start - self.context_end
        assert 0 <= bmin < bmax <= r_end


        if bmin == 0 and bmax >= r_start:
            if bmax == r_end:
                raise NotFullyVisible(True)
            if not self.at_begin:
                raise NotFullyVisible(False)

        if bmax < r_start: return False
        if bmax == r_end and not self.at_end: return False
        return True


    @property
    def frozen(self):
        return self._frozen

    @property
    def at_begin(self):
        return self.pred is None

    @property
    def at_end(self):
        return self.succ is None

    @property
    def overlap(self):
        return self.size - self.stride

    @property
    def context_start(self):
        return 0 if self.at_begin else self.context

    @property
    def context_end(self):
        return 0 if self.at_end else self.context

    def __repr__(self):
        shared  = f'{self.start:03}:{self.end:03}'
        shared += f', size={self.context_start}-{self.size-self.context_start-self.context_end}-{self.context_end}'
        if self.at_end:
            return f'{self.__class__.__name__}({shared})'
        else:
            return f'{self.__class__.__name__}({shared}, overlap={self.overlap}/{self.overlap-self.context_start-self.context_end})'

    @property
    def chain(self):
        blocks = [self]
        while not blocks[-1].at_end:
            blocks.append(blocks[-1].succ)
        return blocks

    def __iter__(self):
        return iter(self.chain)


    @staticmethod
    def cover(size, block_size, min_overlap, context, grid=1, verbose=True):
        """Return chain of grid-aligned blocks to cover the interval [0,size].

        Parameters block_size, min_overlap, and context will be used
        for all blocks of the chain. Only the size of the last block
        may differ.

        Except for the last block, start and end positions of all blocks will
        be multiples of grid. To that end, the provided block parameters may
        be increased to achieve that.

        Note that parameters must be chosen such that the write regions of only
        neighboring blocks are overlapping.

        """
        assert 0 <= min_overlap+2*context < block_size <= size
        assert 0 < grid <= block_size
        block_size = _grid_divisible(grid, block_size, name='block_size', verbose=verbose)
        min_overlap = _grid_divisible(grid, min_overlap, name='min_overlap', verbose=verbose)
        context = _grid_divisible(grid, context, name='context', verbose=verbose)

        size_orig = size
        size = _grid_divisible(grid, size, name='size', verbose=False)

        assert all(v % grid == 0 for v in (size, block_size, min_overlap, context))
        size //= grid
        block_size //= grid
        min_overlap //= grid
        context //= grid

        t = first = Block(block_size, min_overlap, context, None)
        while t.end < size:
            t = t.add_succ()
        last = t

        excess = last.end - size
        t = first
        while excess > 0:
            t.decrease_stride(1)
            excess -= 1
            t = t.succ
            if (t == last): t = first

        if grid > 1:
            size *= grid
            block_size *= grid
            min_overlap *= grid
            context *= grid
            #
            _t = _first = first
            t = first = Block(block_size, min_overlap, context, None)
            t.stride = _t.stride*grid
            while not _t.at_end:
                _t = _t.succ
                t = t.add_succ()
                t.stride = _t.stride*grid
            last = t

            size_delta = size - size_orig
            last.size -= size_delta
            assert 0 <= size_delta < grid

        first.freeze()

        blocks = first.chain

        assert first.start == 0 and last.end == size_orig
        assert all(t.overlap-2*context >= min_overlap for t in blocks if t != last)
        assert all(t.start % grid == 0 and t.end % grid == 0 for t in blocks if t != last)

        if len(blocks) >= 3:
            for t in blocks[:-2]:
                assert t.slice_write.stop <= t.succ.succ.slice_write.start

        return blocks



class BlockND:
    """N-dimensional block.

    Each BlockND simply consists of a 1-dimensional Block per axis and also
    has an id (which should be unique). The n-dimensional region represented
    by each BlockND is the intersection of all 1D Blocks per axis.

    Also see `Block`.

    """
    def __init__(self, id, blocks, axes):
        self.id = id
        self.blocks = tuple(blocks)
        self.axes = axes_check_and_normalize(axes, length=len(self.blocks))
        self.axis_to_block = dict(zip(self.axes,self.blocks))

    def blocks_for_axes(self, axes=None):
        axes = self.axes if axes is None else axes_check_and_normalize(axes)
        return tuple(self.axis_to_block[a] for a in axes)

    def slice_read(self, axes=None):
        return tuple(t.slice_read for t in self.blocks_for_axes(axes))

    def slice_crop_context(self, axes=None):
        return tuple(t.slice_crop_context for t in self.blocks_for_axes(axes))

    def slice_write(self, axes=None):
        return tuple(t.slice_write for t in self.blocks_for_axes(axes))

    def read(self, x, axes=None):
        """Read block "read region" from x (numpy.ndarray or similar)"""
        return x[self.slice_read(axes)]

    def crop_context(self, labels, axes=None):
        return labels[self.slice_crop_context(axes)]

    def write(self, x, labels, axes=None):
        """Write (only entries > 0 of) labels to block "write region" of x (numpy.ndarray or similar)"""
        s = self.slice_write(axes)
        mask = labels > 0
        region = x[s]
        region[mask] = labels[mask]
        x[s] = region

    def is_responsible(self, slices, axes=None):
        return all(t.is_responsible((s.start,s.stop)) for t,s in zip(self.blocks_for_axes(axes),slices))

    def __repr__(self):
        slices =  ','.join(f'{a}={t.start:03}:{t.end:03}' for t,a in zip(self.blocks,self.axes))
        return f'{self.__class__.__name__}({self.id}|{slices})'

    def __iter__(self):
        return iter(self.blocks)

    # ------------------------

    def filter_objects(self, labels, polys, axes=None):
        """Filter out objects that block is not responsible for.

        Given label image 'labels' and dictionary 'polys' of polygon/polyhedron objects,
        only retain those objects that this block is responsible for.

        This function will return a pair (labels, polys) of the modified label image and dictionary.
        It will raise a RuntimeError if an object is found in the overlap area
        of neighboring blocks that violates the assumption to be smaller than 'min_overlap'.

        If parameter 'polys' is None, only the filtered label image will be returned.

        Notes
        -----
        - Important: It is assumed that the object label ids in 'labels' and
          the entries in 'polys' are sorted in the same way.
        - Does not modify 'labels' and 'polys', but returns modified copies.

        Example
        -------
        >>> labels, polys = model.predict_instances(block.read(img))
        >>> labels = block.crop_context(labels)
        >>> labels, polys = block.filter_objects(labels, polys)

        """
        assert np.issubdtype(labels.dtype, np.integer)
        ndim = len(self.blocks_for_axes(axes))
        assert ndim in (2,3)
        assert labels.ndim == ndim and labels.shape == tuple(s.stop-s.start for s in self.slice_crop_context(axes))

        labels_filtered = np.zeros_like(labels)
        # problem_ids = []
        for r in regionprops(labels):
            slices = tuple(slice(r.bbox[i],r.bbox[i+labels.ndim]) for i in range(labels.ndim))
            try:
                if self.is_responsible(slices, axes):
                    labels_filtered[slices][r.image] = r.label
            except NotFullyVisible as e:
                shape_object = tuple(s.stop-s.start for s in slices)
                shape_min_overlap = tuple(t.min_overlap for t in self.blocks_for_axes(axes))
                raise RuntimeError(f"Found object of shape {shape_object}, which violates the assumption of being smaller than 'min_overlap' {shape_min_overlap}. Increase 'min_overlap' to avoid this problem.")


        if polys is None:
            return labels_filtered
        else:
            assert isinstance(polys,dict) and any(k in polys for k in COORD_KEYS)
            filtered_labels = np.unique(labels_filtered)
            filtered_ind = [i-1 for i in filtered_labels if i > 0]
            polys_out = {k: (v[filtered_ind] if k in OBJECT_KEYS else v) for k,v in polys.items()}
            for k in COORD_KEYS:
                if k in polys_out.keys():
                    polys_out[k] = self.translate_coordinates(polys_out[k], axes=axes)

        return labels_filtered, polys_out

    def translate_coordinates(self, coordinates, axes=None):
        """Translate local block coordinates (of read region) to global ones based on block position"""
        ndim = len(self.blocks_for_axes(axes))
        assert isinstance(coordinates, np.ndarray) and coordinates.ndim >= 2 and coordinates.shape[1] == ndim
        start = [s.start for s in self.slice_read(axes)]
        shape = tuple(1 if d!=1 else ndim for d in range(coordinates.ndim))
        start = np.array(start).reshape(shape)
        return coordinates + start

    @staticmethod
    def cover(shape, axes, block_size, min_overlap, context, grid=1):
        """Return grid-aligned n-dimensional blocks to cover region
        of the given shape with axes semantics.

        Parameters block_size, min_overlap, and context can be different per
        dimension/axis (if provided as list) or the same (if provided as
        scalar value).

        Also see `Block.cover`.

        """
        shape = tuple(shape)
        n = len(shape)
        axes = axes_check_and_normalize(axes, length=n)
        if np.isscalar(block_size):  block_size  = n*[block_size]
        if np.isscalar(min_overlap): min_overlap = n*[min_overlap]
        if np.isscalar(context):     context     = n*[context]
        if np.isscalar(grid):        grid        = n*[grid]
        assert n == len(block_size) == len(min_overlap) == len(context) == len(grid)
        cover_1d = [Block.cover(*args) for args in zip(shape, block_size, min_overlap, context, grid)]
        return tuple(BlockND(i,blocks,axes) for i,blocks in enumerate(product(*cover_1d)))



class Polygon:

    def __init__(self, coord, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox(coord, shape_max=shape_max) if bbox is None else bbox
        self.coord = coord - np.array([r[0] for r in self.bbox]).reshape(2,1)
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1]-r[0] for r in self.bbox)
        rr,cc = polygon(*self.coord, self.shape)
        self.mask = np.zeros(self.shape, bool)
        self.mask[rr,cc] = True

    @staticmethod
    def coords_bbox(*coords, shape_max=None):
        assert all(isinstance(c, np.ndarray) and c.ndim==2 and c.shape[0]==2 for c in coords)
        if shape_max is None:
            shape_max = (np.inf, np.inf)
        coord = np.concatenate(coords, axis=1)
        mins = np.maximum(0,         np.floor(np.min(coord,axis=1))).astype(int)
        maxs = np.minimum(shape_max, np.ceil (np.max(coord,axis=1))).astype(int)
        return tuple(zip(tuple(mins),tuple(maxs)))



class Polyhedron:

    def __init__(self, dist, origin, rays, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox((dist, origin), rays=rays, shape_max=shape_max) if bbox is None else bbox
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1]-r[0] for r in self.bbox)
        _origin = origin.reshape(1,3) - np.array([r[0] for r in self.bbox]).reshape(1,3)
        self.mask = polyhedron_to_label(dist[np.newaxis], _origin, rays, shape=self.shape, verbose=False).astype(bool)

    @staticmethod
    def coords_bbox(*dist_origin, rays, shape_max=None):
        dists, points = zip(*dist_origin)
        assert all(isinstance(d, np.ndarray) and d.ndim==1 and len(d)==len(rays) for d in dists)
        assert all(isinstance(p, np.ndarray) and p.ndim==1 and len(p)==3 for p in points)
        dists, points, verts = np.stack(dists)[...,np.newaxis], np.stack(points)[:,np.newaxis], rays.vertices[np.newaxis]
        coord = dists * verts + points
        coord = np.concatenate(coord, axis=0)
        if shape_max is None:
            shape_max = (np.inf, np.inf, np.inf)
        mins = np.maximum(0,         np.floor(np.min(coord,axis=0))).astype(int)
        maxs = np.minimum(shape_max, np.ceil (np.max(coord,axis=0))).astype(int)
        return tuple(zip(tuple(mins),tuple(maxs)))

def predict_big(model, *args, **kwargs):
    from .models import StarDist2D, StarDist3D
    if isinstance(model,(StarDist2D,StarDist3D)):
        dst = model.__class__.__name__
    else:
        dst = '{StarDist2D, StarDist3D}'
    raise RuntimeError(f"This function has moved to {dst}.predict_instances_big.")

class NotFullyVisible(Exception):
    pass

def _grid_divisible(grid, size, name=None, verbose=True):
    if size % grid == 0:
        return size
    _size = size
    size = math.ceil(size / grid) * grid
    if bool(verbose):
        print(f"{verbose if isinstance(verbose,str) else ''}increasing '{'value' if name is None else name}' from {_size} to {size} to be evenly divisible by {grid} (grid)", flush=True)
    assert size % grid == 0
    return size