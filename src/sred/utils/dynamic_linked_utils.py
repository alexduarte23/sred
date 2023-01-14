from typing import Optional, Union
import numpy as np
import ctypes
import pathlib
import platform

if platform.uname()[0] == 'Windows':
    _dllsuffix = '.dll'
elif platform.uname()[0] == 'Linux':
    _dllsuffix = '.so'
else:
    _dllsuffix = '.dylib'

_dllpath = pathlib.Path(__file__).parent / ("fast_utils/shared/fast_utils" + _dllsuffix)

_mydll = ctypes.cdll.LoadLibrary(str(_dllpath))


def setDLL(path: Union[str, pathlib.Path]) -> None:
    global _mydll, _dllpath, _dllsuffix

    dllpath = pathlib.Path(path)
    if not dllpath.is_file():
        raise RuntimeError('File not found.')
    if dllpath.suffix != _dllsuffix:
        raise RuntimeError('Bad file extension, only .dll suported.')

    _mydll = ctypes.cdll.LoadLibrary(str(dllpath))
    _dllpath = dllpath

def getDLL():
    return _dllpath

def hello_test():
    _mydll.hello_test()


# REGISTRATION

class CAMPARAMS(ctypes.Structure):
    _fields_ = [
        ('d_fx', ctypes.c_double),
        ('d_fy', ctypes.c_double),
        ('d_cx', ctypes.c_double),
        ('d_cy', ctypes.c_double),
        ('rgb_fx', ctypes.c_double),
        ('rgb_fy', ctypes.c_double),
        ('rgb_cx', ctypes.c_double),
        ('rgb_cy', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('t_x', ctypes.c_double),
        ('t_y', ctypes.c_double)
    ]



def hole_interpolation(d_img: np.ndarray) -> np.ndarray:
    """Applies 2-step interpolation to fill depth holes

    Parameters
    ----------
    d_img : ndarray
        Depth image with holes

    Returns
    -------
    ndarray
        New depth image with holes filled through interpolation
    """

    if d_img.dtype != 'uint16':
        print('Unsupported image dtype')
        return
    if not (len(d_img.shape) == 3 and d_img.shape[-1] == 1 or len(d_img.shape) == 2):
        print('Unsupported image shape')
        return

    out_img = np.zeros_like(d_img)

    c_data_ptr = d_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_width = ctypes.c_uint32(d_img.shape[1])
    c_height = ctypes.c_uint32(d_img.shape[0])
    _mydll.hole_interpolation_api(c_data_ptr, c_width, c_height, c_out_ptr)

    return out_img

def smoothen_filled_holes(d_img: np.ndarray, filled_img: np.ndarray) -> np.ndarray:
    """Applies box-blur on interpolated holes

    Parameters
    ----------
    d_img : ndarray
        Depth image with holes
    filled_img : ndarray
        Depth image with holes filled through interpolation

    Returns
    -------
    ndarray
        New depth image with smoother interpolated holes
    """

    if d_img.dtype != 'uint16':
        print('Unsupported image dtype')
        return
    if not (len(d_img.shape) == 3 and d_img.shape[-1] == 1 or len(d_img.shape) == 2):
        print('Unsupported image shape')
        return

    out_img = np.zeros_like(d_img)

    c_data_ptr = d_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_filled_ptr = filled_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_width = ctypes.c_uint32(d_img.shape[1])
    c_height = ctypes.c_uint32(d_img.shape[0])
    _mydll.smoothen_filled_holes_api(c_data_ptr, c_filled_ptr, c_width, c_height, c_out_ptr)

    return out_img


def register_rgb(rgb_img: np.ndarray, d_img: np.ndarray, cam_params: dict) -> np.ndarray:
    """Registers rgb data (rgb -> d)

    Parameters
    ----------
    rgb_img : ndarray
        RGB image
    d_img : ndarray
        Depth image
    cam_params : dict
        Intrinsic and extrinsic parameters of the RGB-D camera

    Returns
    -------
    ndarray
        New rgb image registered to the depth image
    """

    if d_img.dtype != 'uint16' or rgb_img.dtype != 'uint8':
        print('Unsupported image dtype')
        return
    if not (len(d_img.shape) == 3 and d_img.shape[-1] == 1 or len(d_img.shape) == 2):
        print('Unsupported image shape')
        return
    if len(rgb_img.shape) != 3 or rgb_img.shape[-1] != 3:
        print('Unsupported image shape')
        return

    out_img = np.zeros(d_img.shape+(3,), dtype=np.uint8)

    c_params_ptr = ctypes.pointer(CAMPARAMS(**cam_params))

    c_d_data_ptr = d_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_rgb_data_ptr = rgb_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    c_out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    c_d_width = ctypes.c_uint32(d_img.shape[1])
    c_d_height = ctypes.c_uint32(d_img.shape[0])
    c_rgb_width = ctypes.c_uint32(rgb_img.shape[1])
    c_rgb_height = ctypes.c_uint32(rgb_img.shape[0])
    _mydll.register_rgb_api(
        c_rgb_data_ptr, c_rgb_width, c_rgb_height,
        c_d_data_ptr, c_d_width, c_d_height,
        c_params_ptr,
        c_out_ptr)

    return out_img

def register_d(rgb_img: np.ndarray, d_img: np.ndarray, cam_params: dict) -> np.ndarray:
    """Registers depth data (d -> rgb)

    Parameters
    ----------
    rgb_img : ndarray
        RGB image
    d_img : ndarray
        Depth image
    cam_params : dict
        Intrinsic and extrinsic parameters of the RGB-D camera

    Returns
    -------
    ndarray
        New depth image registered to the RGB image
    """

    if d_img.dtype != 'uint16' or rgb_img.dtype != 'uint8':
        print('Unsupported image dtype')
        return
    if not (len(d_img.shape) == 3 and d_img.shape[-1] == 1 or len(d_img.shape) == 2):
        print('Unsupported image shape')
        return
    if len(rgb_img.shape) != 3 or rgb_img.shape[-1] != 3:
        print('Unsupported image shape')
        return

    out_img = np.zeros(rgb_img.shape[:-1], dtype=np.uint16)

    c_params_ptr = ctypes.pointer(CAMPARAMS(**cam_params))

    c_d_data_ptr = d_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_rgb_data_ptr = rgb_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    c_out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_d_width = ctypes.c_uint32(d_img.shape[1])
    c_d_height = ctypes.c_uint32(d_img.shape[0])
    c_rgb_width = ctypes.c_uint32(rgb_img.shape[1])
    c_rgb_height = ctypes.c_uint32(rgb_img.shape[0])
    _mydll.register_d_api(
        c_rgb_data_ptr, c_rgb_width, c_rgb_height,
        c_d_data_ptr, c_d_width, c_d_height,
        c_params_ptr,
        c_out_ptr)

    return out_img


# INPAINTING

def inpaint(img: np.ndarray, mask: np.ndarray, guide: Optional[np.ndarray] = None, radius: float = 5) -> np.ndarray:
    """Applies FMM inpainting, or GFMM inpainting if a guide is provided

    Parameters
    ----------
    img : ndarray
        Image to be inpainted (2-D or 3-D)
    mask : ndarray
        Mask of pixels to inpaint (2-D)
    guide : ndarray
        Optional guiding image (2-D or 3-D)
    radius : float
        Radius of effect during pixel inpainting 

    Returns
    -------
    ndarray
        New inpainted image
    """

    if not isinstance(img, np.ndarray):
        print('Image must be a numpy array')
        return
    if not np.issubdtype(img.dtype, np.number):
        print('Unsupported image dtype')
        return
    if not (len(img.shape) == 3 and img.shape[-1] == 1
            or len(img.shape) == 3 and img.shape[-1] == 3
            or len(img.shape) == 2):
        print('Unsupported image shape')
        return
    if not isinstance(mask, np.ndarray):
        print('Mask must be a numpy array')
        return
    if not (len(mask.shape) == 3 and mask.shape[-1] == 1 or len(mask.shape) == 2):
        print('Unsupported mask shape')
        return
    if guide is not None:
        if not isinstance(guide, np.ndarray):
            print('Guide must be a numpy array')
            return
        if not np.issubdtype(guide.dtype, np.number):
            print('Unsupported guide dtype')
            return
        if not (len(guide.shape) == 3 and guide.shape[-1] == 1
                or len(guide.shape) == 3 and guide.shape[-1] == 3
                or len(guide.shape) == 2):
            print('Unsupported guide shape')
            return
        if guide.shape[:2] != img.shape[:2]:
            print('Mismatch between guide shape and image shape')
            return
    if not isinstance(radius, int) and not isinstance(radius, float) or radius < 1:
        print('Invalid radius value')
        return


    img_type = img.dtype
    img = img.astype(np.float64)
    mask = (mask > 0).astype(np.uint8)

    out_img = np.zeros(img.shape, dtype=np.float64)

    c_img_data_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_mask_data_ptr = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    c_width = ctypes.c_uint32(img.shape[1])
    c_height = ctypes.c_uint32(img.shape[0])
    c_channels = ctypes.c_uint8(1 if len(img.shape)==2 else img.shape[2])
    c_radius = ctypes.c_double(radius)
    c_out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if guide is None:
        _mydll.inpaint_FMM_api(
            c_img_data_ptr, c_width, c_height, c_channels,
            c_mask_data_ptr,
            c_radius,
            c_out_ptr)
    else:
        guide = guide.astype(np.float64)
        c_guide_data_ptr = guide.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_guide_channels = ctypes.c_uint8(1 if len(guide.shape)==2 else guide.shape[2])

        _mydll.inpaint_GFMM_api(
            c_img_data_ptr, c_width, c_height, c_channels,
            c_mask_data_ptr,
            c_guide_data_ptr, c_guide_channels,
            c_radius,
            c_out_ptr)

    out_img = out_img.astype(img_type)

    return out_img


# MISC

def cleanup_depth(d_img: np.ndarray, min_size: int) -> np.ndarray:
    """Removes pixel regions with size lesser than the minimum size allowed

    Parameters
    ----------
    d_img : ndarray
        Depth image
    min_size : float
        Minimum size a contiguous pixel regions must have to be kept

    Returns
    -------
    ndarray
        New depth image with regions smaller than min_size converted to holes
    """

    if not isinstance(d_img, np.ndarray):
        print('Depth image must be a numpy array')
        return
    if d_img.dtype != 'uint16':
        print('Unsupported image dtype')
        return
    if not (len(d_img.shape) == 3 and d_img.shape[-1] == 1 or len(d_img.shape) == 2):
        print('Unsupported image shape')
        return
    if not isinstance(min_size, int) and not isinstance(min_size, float) or min_size <= 0:
        print('Invalid type or value for min_size')
        return

    out_img = np.empty_like(d_img)

    c_d_data_ptr = d_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_out_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_width = ctypes.c_uint32(d_img.shape[1])
    c_height = ctypes.c_uint32(d_img.shape[0])
    c_min_size = ctypes.c_uint32(min_size)

    _mydll.cleanup_depth_api(
        c_d_data_ptr, c_width, c_height,
        c_min_size,
        c_out_ptr)

    return out_img


if __name__ == "__main__":
    print('Loaded lib:', getDLL())
    print('Testing...')
    hello_test()