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


def array_prepare(arr_in: np.ndarray, r: int) -> Iterator[np.ndarray]:
    return (arr_in[sl] for msk in mask_combinations(arr_in.shape, r) for sl in mask_to_slices(msk))


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


if __name__ == '__main__':
    PRINT_ARR = '{arr}, ndim = {arr.ndim}'

    VARS = {'SHAPE': (5,) * 3, 'NDIM': 2, 'W_SHAPE': 3}
    save_arr: np.ndarray = np.arange(np.array(VARS['SHAPE']).prod()).reshape(VARS['SHAPE'])

    print('new array', VARS)
    for x, arr in enumerate(save_arr):
        print(x)
        print(PRINT_ARR.format(arr=arr))

    print('array_prepare', VARS)
    for x, arr in enumerate(array_prepare(save_arr, VARS['NDIM'])):
        print(x)
        print(PRINT_ARR.format(arr=arr))

    print('view_as_windows', VARS)
    for x, arr in enumerate(view_as_windows(save_arr, VARS['W_SHAPE'])):
        print(x)
        print(PRINT_ARR.format(arr=arr))

    print('view_as_blocks', VARS)
    for x, arr in enumerate(view_as_windows(save_arr, VARS['W_SHAPE'], VARS['W_SHAPE'])):
        print(x)
        print(PRINT_ARR.format(arr=arr))
