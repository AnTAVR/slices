from itertools import combinations
from typing import Tuple, Union, Iterator

import numpy as np

__all__ = ['MaskVal', 'Mask', 'SliceVal', 'Slice', 'mask_combinations', 'mask_to_slices', 'array_prepare', 'view_as_windows']

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


def array_prepare(arr_in: np.ndarray, ndim: int) -> Iterator[np.ndarray]:
    return (arr_in[sl] for msk in mask_combinations(arr_in.shape, ndim) for sl in mask_to_slices(msk))


def view_as_windows(arr_in: np.ndarray, win_shape: Union[int, Tuple[int, ...]], step: Union[int, Tuple[int, ...]] = 1) -> np.ndarray:
    ndim: int = arr_in.ndim

    if isinstance(win_shape, int):
        win_shape = (win_shape,) * ndim
    # _len = len(win_shape)
    # assert _len == ndim, (ndim, _len)

    if isinstance(step, int):
        step = (step,) * ndim
    # _len = len(step)
    # assert _len == ndim, (ndim, _len)

    shape: np.ndarray = ((np.array(arr_in.shape) - np.array(win_shape)) // np.array(step)) + 1
    out_shape: Tuple[int, ...] = (shape.prod(),) + win_shape
    shape: Tuple[int, ...] = tuple(shape) + win_shape
    strides: Tuple[int, ...] = arr_in[tuple(slice(None, None, s) for s in step)].strides + arr_in.strides

    return np.lib.stride_tricks.as_strided(arr_in, shape=shape, strides=strides).reshape(out_shape)


if __name__ == '__main__':
    VARS = {}

    PRINT_ARR = '{arr}, ndim = {arr.ndim}'

    VARS['NDIM'] = 2
    VARS['SHAPE'] = (3, 4, 5)

    print('new array')
    print(VARS)
    arr: np.ndarray = np.arange(np.array(VARS['SHAPE']).prod()).reshape(VARS['SHAPE'])
    print(PRINT_ARR.format(arr=arr))

    print('array_prepare')
    print(VARS)
    for arr in array_prepare(arr, VARS['NDIM']):
        print(PRINT_ARR.format(arr=arr))

    VARS['SHAPE'] = (5,) * 3

    print('new array')
    print(VARS)

    arr: np.ndarray = np.arange(np.array(VARS['SHAPE']).prod()).reshape(VARS['SHAPE'])
    print(PRINT_ARR.format(arr=arr))

    save_arr = arr

    print('view_as_windows')
    VARS['W_SHAPE'] = (2,) * len(VARS['SHAPE'])
    print(VARS)
    arr = view_as_windows(save_arr, VARS['W_SHAPE'])
    print(PRINT_ARR.format(arr=arr))

    print('view_as_blocks')
    print(VARS)
    arr = view_as_windows(save_arr, VARS['W_SHAPE'], VARS['W_SHAPE'])
    print(PRINT_ARR.format(arr=arr))
