import itertools
from typing import Tuple, Union, Any, Generator

Mask = Tuple[Union[range, slice], ...]
Slice = Tuple[Union[int, slice], ...]


def get_slice_masks(shape: Tuple[int, ...], ndim: int) -> Generator[Mask, Any, None]:
    _ndim = len(shape)
    # assert 0 <= ndim <= _ndim
    for _comb in itertools.combinations(range(_ndim - 1, -1, -1), ndim):  # type: Tuple[int, ...]
        yield tuple(slice(shape[n]) if n in _comb else range(shape[n]) for n in range(_ndim))


def gen_slices(slice_mask: Mask, prev_slice: Slice = tuple(), dim: int = 0) -> Generator[Slice, Any, None]:
    _ndim = len(slice_mask)
    # assert 0 <= dim <= _ndim + 1
    if dim >= _ndim:
        yield prev_slice
        return

    _range = slice_mask[dim]  # type: Union[range, slice]
    next_dim = dim + 1

    if isinstance(_range, slice):
        yield from gen_slices(slice_mask, prev_slice + (_range,), next_dim)
        return

    for _i in _range:
        yield from gen_slices(slice_mask, prev_slice + (_i,), next_dim)


if __name__ == '__main__':
    import functools
    import operator
    import numpy as np

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
