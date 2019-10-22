from itertools import combinations
from typing import Tuple, Union, Any, Generator

MaskVal = Union[range, slice]
Mask = Tuple[MaskVal, ...]
SliceVal = Union[int, slice]
Slice = Tuple[SliceVal, ...]


def get_slice_masks(shape: Tuple[int, ...], ndim: int) -> Generator[Mask, Any, None]:
    _len = len(shape)
    # assert 0 <= ndim <= _len, (ndim, _len)
    for _comb in combinations(range(_len - 1, -1, -1), ndim):  # type: Tuple[int, ...]
        yield tuple(slice(shape[_n]) if _n in _comb else range(shape[_n]) for _n in range(_len))


def gen_slices(slice_mask: Mask, prev_slice: Slice = tuple(), ndim: int = 0) -> Generator[Slice, Any, None]:
    _len = len(slice_mask)
    # assert 0 <= ndim <= _len, (ndim, _len)
    if ndim == _len:
        _slice = prev_slice  # type: Slice
        yield _slice
        return

    _ndim = ndim + 1
    _mask_val = slice_mask[ndim]  # type: MaskVal

    if isinstance(_mask_val, slice):
        _slice = prev_slice + (_mask_val,)  # type: Slice
        yield from gen_slices(slice_mask, _slice, _ndim)
        return

    for _n in _mask_val:
        _slice = prev_slice + (_n,)  # type: Slice
        yield from gen_slices(slice_mask, _slice, _ndim)


if __name__ == '__main__':
    import functools
    import operator
    import numpy as np

    SHAPE = (3, 4, 5)
    NDIM = 2

    z = np.arange(functools.reduce(operator.mul, SHAPE)).reshape(SHAPE)  # type: np.ndarray
    print(z)

    for mask in get_slice_masks(z.shape, NDIM):
        print('mask =', mask)
        for sl in gen_slices(mask):
            print('z[', sl, ']', sep='')
            r = z[sl]  # type: np.ndarray
            print(r)
