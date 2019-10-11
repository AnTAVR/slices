import functools
import itertools
import operator
from typing import Tuple, Union, Any, Generator

import numpy as np

Mask = Tuple[Union[range, slice], ...]
Slice = Tuple[Union[int, slice], ...]


def get_slice_masks(shape: Tuple[int, ...], ndim: int) -> Generator[Mask, Any, None]:
    _ndim = len(shape)
    # assert 0 <= ndim <= _ndim
    for _comb in itertools.combinations(range(_ndim - 1, -1, -1), ndim):  # type: Tuple[int, ...]
        yield tuple(slice(shape[n]) if n in _comb else range(shape[n]) for n in range(_ndim))


def gen_slices(slice_mask: Mask, tmp_slice: Slice = tuple(), ndim: int = 0) -> Generator[Slice, Any, None]:
    _ndim = len(slice_mask)
    # assert 0 <= ndim <= _ndim + 1
    if ndim >= _ndim:
        yield tmp_slice
        return

    _range = slice_mask[ndim]  # type: Union[range, slice]
    ndim += 1
    if isinstance(_range, slice):
        yield from gen_slices(slice_mask, tmp_slice + (_range,), ndim)
    else:
        for _i in _range:
            yield from gen_slices(slice_mask, tmp_slice + (_i,), ndim)


if __name__ == '__main__':
    SHAPE = (3, 2, 4)
    NDIM = 2

    z = np.arange(functools.reduce(operator.mul, SHAPE)).reshape(SHAPE)  # type: np.ndarray
    print(z)

    for mask in get_slice_masks(z.shape, NDIM):
        print('mask =', mask)
        for sl in gen_slices(mask):
            print('z[', sl, ']', sep='')
            r = z[sl]  # type: np.ndarray
            print(r)
