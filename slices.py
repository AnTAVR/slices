from itertools import combinations
from typing import Tuple, Union, Iterator

import numpy as np

__all__ = ['MaskVal', 'Mask', 'SliceVal', 'Slice', 'mask_combinations', 'mask_to_slices', 'view_as_windows']

MaskVal = Union[range, slice]
Mask = Tuple[MaskVal, ...]
SliceVal = Union[int, slice]
Slice = Tuple[SliceVal, ...]


def mask_combinations(shape: Tuple[int, ...], r: int) -> Iterator[Mask]:
    _len = len(shape)
    # assert 0 <= r <= _len, (r, _len)
    for comb_dims in combinations(range(_len - 1, -1, -1), r):  # type: Tuple[int, ...]
        yield tuple(slice(shape[dim]) if dim in comb_dims else range(shape[dim]) for dim in range(_len))


def mask_to_slices(mask: Mask) -> Iterator[Slice]:
    _len = len(mask)

    def _mts(_slice: Slice, dim: int) -> Iterator[Slice]:
        # assert 0 <= dim <= _len, (dim, _len)
        if dim == _len:
            yield _slice
            return

        value = mask[dim]  # type: MaskVal
        dim += 1

        if isinstance(value, slice):
            yield from _mts(_slice + (value,), dim)
            return

        for n in value:
            yield from _mts(_slice + (n,), dim)

    return _mts(tuple(), 0)


def view_as_windows(arr_in: np.ndarray, shape: Union[int, Tuple[int, ...]], step: Union[int, Tuple[int, ...]] = 1) -> np.ndarray:
    ndim: int = arr_in.ndim

    if isinstance(shape, int):
        shape = (shape,) * ndim
    # _len = len(shape)
    # assert _len == ndim, (ndim, _len)

    if isinstance(step, int):
        step = (step,) * ndim
    # _len = len(step)
    # assert _len == ndim, (ndim, _len)

    indices_shape: np.ndarray = ((np.array(arr_in.shape) - np.array(shape)) // np.array(step)) + 1
    new_shape = tuple(indices_shape) + shape

    slices: Slice = tuple(slice(None, None, st) for st in step)
    new_strides: Tuple[int, ...] = arr_in[slices].strides + arr_in.strides

    arr_out = np.lib.stride_tricks.as_strided(arr_in, shape=new_shape, strides=new_strides)
    return arr_out


if __name__ == '__main__':
    import functools
    import operator

    SHAPE = (3, 4, 5)
    NDIM = 2

    arr: np.ndarray = np.arange(functools.reduce(operator.mul, SHAPE)).reshape(SHAPE)
    print(arr, end='\n\n')

    print('mask_combinations')
    for msk in mask_combinations(arr.shape, NDIM):
        print('mask =', msk)
        for sl in mask_to_slices(msk):
            print('z[', sl, ']', sep='')
            out_arr: np.ndarray = arr[sl]
            print(out_arr, end='\n\n')

    SHAPE = (4, 5)
    W_SHAPE = (2, 2)

    arr = np.arange(functools.reduce(operator.mul, SHAPE)).reshape(SHAPE)  # type: np.ndarray
    print(arr, end='\n\n')

    print('view_as_windows')
    for out_arr in view_as_windows(arr, W_SHAPE):
        print(out_arr, end='\n\n')

    print('view_as_blocks')
    for out_arr in view_as_windows(arr, W_SHAPE, W_SHAPE):
        print(out_arr, end='\n\n')
