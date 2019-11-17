from itertools import combinations, product
from math import ceil
from typing import Tuple, Union, Iterator, Optional

import numpy as np

__all__ = ['MaskVal', 'Mask', 'SliceVal', 'Slice', 'mask_combinations', 'mask_to_slices', 'array_prepare', 'view_as_windows']

MaskVal = Union[range, slice]
Mask = Tuple[MaskVal, ...]
SliceVal = Union[int, slice]
Slice = Tuple[SliceVal, ...]


def mask_combinations(shape: Tuple[int, ...], ndim: int) -> Iterator[Mask]:
    _len = len(shape)
    # assert 0 <= ndim <= _len, (ndim, _len)
    for comb_dims in combinations(range(_len - 1, -1, -1), ndim):  # type: Tuple[int, ...]
        yield tuple(slice(val) if dim in comb_dims else range(val) for dim, val in enumerate(shape))


def mask_to_slices(mask: Mask) -> Iterator[Slice]:
    _len = len(mask)

    def _mts(new_slice: Slice, dim: int) -> Iterator[Slice]:
        # assert 0 <= dim <= _len, (dim, _len)
        if dim == _len:
            yield new_slice
            return

        value = mask[dim]  # type: MaskVal
        dim += 1

        if isinstance(value, slice):
            sl = new_slice + (value,)
            yield from _mts(sl, dim)
            return

        for val in value:
            sl = new_slice + (val,)
            yield from _mts(sl, dim)

    return _mts(tuple(), 0)


def array_prepare(arr_in: np.ndarray, ndim: int) -> Iterator[np.ndarray]:
    return (arr_in[sl] for msk in mask_combinations(arr_in.shape, ndim) for sl in mask_to_slices(msk))


def view_as_windows(arr_in: np.ndarray,
                    win_shape: Union[int, Tuple[int, ...]],
                    step: Union[int, Tuple[int, ...]] = 1) -> np.ndarray:
    ndim: int = arr_in.ndim

    if isinstance(win_shape, int):
        win_shape = (win_shape,) * ndim
    # _len = len(win_shape)
    # assert _len == ndim, (ndim, _len, win_shape)

    if isinstance(step, int):
        step = (step,) * ndim
    # _len = len(step)
    # assert _len == ndim, (ndim, _len, step)

    indices_shape: np.ndarray = ((np.array(arr_in.shape) - np.array(win_shape)) // np.array(step)) + 1

    shape: Tuple[int, ...] = tuple(indices_shape) + win_shape
    strides: Tuple[int, ...] = arr_in[tuple(slice(None, None, s) for s in step)].strides + arr_in.strides
    out_shape: Tuple[int, ...] = (indices_shape.prod(),) + win_shape

    return np.lib.stride_tricks.as_strided(arr_in, shape, strides).reshape(out_shape)


def array_ext(arr_in: np.ndarray, len_line: int) -> Optional[np.ndarray]:
    r = 2
    cl: int = ceil(len_line / r)

    ret = None
    row = None
    i = 0
    for s in product((slice(-cl, None), slice(None, None), slice(None, cl)), repeat=r):  # type: Tuple[slice, ...]
        val: np.ndarray = arr_in[s]
        if i <= r:
            row = val if row is None else np.hstack((row, val))
            i += 1
        else:
            ret = row if ret is None else np.vstack((ret, row))
            i, row = 1, val
    else:
        if ret is not None:
            ret = np.vstack((ret, row))
    return ret


if __name__ == '__main__':
    PRINT_ARR = '{arr}, ndim = {arr.ndim}'

    VARS = {'SHAPE': (5,) * 2, 'NDIM': 2, 'W_SHAPE': 3}
    save_arr: np.ndarray = np.arange(np.array(VARS['SHAPE']).prod()).reshape(VARS['SHAPE'])

    print('new array', VARS)
    arr = save_arr
    print(PRINT_ARR.format(arr=arr))

    print('array_prepare', VARS)
    for x, arr in enumerate(array_prepare(save_arr, VARS['NDIM'])):
        print(x)
        print(PRINT_ARR.format(arr=arr))

    print('view_as_windows', VARS)
    arr = view_as_windows(save_arr, VARS['W_SHAPE'])
    print(PRINT_ARR.format(arr=arr))

    print('view_as_blocks', VARS)
    arr = view_as_windows(save_arr, VARS['W_SHAPE'], VARS['W_SHAPE'])
    print(PRINT_ARR.format(arr=arr))

    print('array_ext', VARS)
    arr = array_ext(save_arr, VARS['W_SHAPE'])
    print(PRINT_ARR.format(arr=arr))
