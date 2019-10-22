from itertools import combinations
from typing import Tuple, Union, Any, Generator

MaskVal = Union[range, slice]
Mask = Tuple[MaskVal, ...]
SliceVal = Union[int, slice]
Slice = Tuple[SliceVal, ...]


def slice_masks(shape: Tuple[int, ...], ndim: int) -> Generator[Mask, Any, None]:
    _len = len(shape)
    # assert 0 <= ndim <= _len, (ndim, _len)
    for comb_dims in combinations(range(_len - 1, -1, -1), ndim):  # type: Tuple[int, ...]
        yield tuple(slice(shape[dim]) if dim in comb_dims else range(shape[dim]) for dim in range(_len))


def mask_to_slices(slice_mask: Mask) -> Generator[Slice, Any, None]:
    _len = len(slice_mask)

    def _mts(_slice: Slice, dim: int):
        # assert 0 <= dim <= _len, (dim, _len)
        if dim == _len:
            yield _slice
            return

        value = slice_mask[dim]  # type: MaskVal
        dim += 1

        if isinstance(value, slice):
            yield from _mts(_slice + (value,), dim)
            return

        for n in value:
            yield from _mts(_slice + (n,), dim)

    return _mts(tuple(), 0)


if __name__ == '__main__':
    import functools
    import operator
    import numpy as np

    SHAPE = (3, 4, 5)
    NDIM = 2

    z = np.arange(functools.reduce(operator.mul, SHAPE)).reshape(SHAPE)  # type: np.ndarray
    print(z)

    for mask in slice_masks(z.shape, NDIM):
        print('mask =', mask)
        for sl in mask_to_slices(mask):
            print('z[', sl, ']', sep='')
            r = z[sl]  # type: np.ndarray
            print(r)
