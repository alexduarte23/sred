from typing import Optional, Union
import cv2
import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import tkinter as tk


def read_img(source: Union[str, pathlib.Path, np.ndarray, list, tuple], verbose: Optional[bool] = False) -> np.ndarray:
    """Reads depth and RGB images from disk to numpy arrays

    If numpy array is given, the same object is returned.
    If a list or tuple is given, a numpy array is returned.
    Single channel images are always returned as 2-D arrays.
    NOTE: if using string, best to use r-string

    Parameters
    ----------
    source : str, pathlib.Path, list, tuple
        Source to read or normalize format
    
    verbose : bool
        Control verbosity of I/O and formatting process

    Returns
    -------
    ndarray
        Image data
    """
    
    if isinstance(source, str) or isinstance(source, pathlib.Path):
        filename = pathlib.Path(source)
        data = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
        if len(data.shape) == 3:
            if data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            elif data.shape[2] == 4:
                data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA)

    elif isinstance(source, list) or isinstance(source, tuple):
        data = np.array(source)
    elif isinstance(source, np.ndarray):
        data = source
    else:
        raise TypeError('source must be a string, path, numpy array, list or tuple')
    
    if not np.issubdtype(data.dtype, np.floating) and not np.issubdtype(data.dtype, np.integer):
        raise TypeError(f'image data type must be a flavour of float or int. {data.dtype} found')

    if len(data.shape) == 3:
        if data.shape[2] == 1:
            data = data[:,:,0]
            if verbose: print(f'Warning: grayscale image w/ dedicated channel, reshaped to {data.shape}')
        elif data.shape[2] == 3:
            if verbose: print('Notice: RGB image')
        elif data.shape[2] == 4:
            data = data[:,:,:3]
            if verbose: print(f'Warning: image with alpha channel, reshaped to RGB {data.shape}')
        else:
            raise ValueError(f'unsupported number of channels ({data.shape[2]}), image must be grayscale or RGB')
    elif len(data.shape) < 2 or len(data.shape) > 3:
        raise ValueError(f'image array has improper number of dimensions ({len(data.shape)})')

    return data


def show(source: Union[str, pathlib.Path, np.ndarray, list, tuple]) -> None:
    """Display RGB and depth images

    NOTE: if using string, best to use r-string

    Parameters
    ----------
    source : str, pathlib.Path, list, tuple
        Source to read or normalize format

    Returns
    -------
    ndarray
        Image data
    """

    image = read_img(source)

    if np.issubdtype(image.dtype, np.floating):
        vmin = 0
        vmax = 1
    else: # np.integer
        vmin = np.iinfo(image.dtype).min
        vmax = np.iinfo(image.dtype).max

    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


def write_img(image: np.ndarray, filename: Optional[Union[str, pathlib.Path]] = None):
    """Save image to disk

    Forces PNG extension
    If filename is None or empty, system file explorer dialog is opened

    Parameters
    ----------
    image : ndarray
        Image to save
    
    filename : str, pathlib.Path
        Pathname where the image is store, or None to prompt OS dialog window

    Returns
    -------
    pathlib.Path
        Path to where image was stored
    """

    if filename == None or filename == '':
        root = tk.Tk()
        #root.overrideredirect(True)
        root.attributes('-alpha', 0)
        root.lift()
        root.attributes('-topmost', True)
        types = [('PNG Image', '*.png')]
        filename = tk.filedialog.asksaveasfilename(filetypes=types, defaultextension=types, initialdir='.', initialfile='untitled_image')
        root.destroy()
        if filename == '':
            print('Save operation canceled.')
            return None

    filename = pathlib.Path(filename)
    if filename.suffix != '.png':
        filename = filename.parent / (filename.stem + '.png')

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(filename), image)

    return filename
    

def show_hist(image: np.ndarray) -> None:
    """Display histogram of depth image

    Parameters
    ----------
    image : ndarray
        Depth image
    """

    if len(image.shape) != 2:
        raise ValueError('not a depth image, image must be a 2d array')

    plt.hist(image.flatten(), bins='auto', color = "skyblue", ec='black')
    plt.show()


def get_stats(image: np.ndarray, clipping_range: Optional[tuple] = None) -> dict:
    """Compute statistics of depth image

    Parameters
    ----------
    image : ndarray
        Depth image
    
    clipping_range : tuple
        Range for clipping stats

    Returns
    -------
    dict
        Computed stats
    """

    total_points = image.shape[0] * image.shape[1]
    min_val = np.min(image[image>0])
    max_val = np.max(image)
    holes = np.count_nonzero(image == 0)
    mean = np.mean(image, where=image>0)
    std = np.std(image, where=image>0)

    if clipping_range == None:
        clip_min = clip_max = 0
    else:
        clip_min = np.count_nonzero((image > 0) & (image < clipping_range[0]))
        clip_max = np.count_nonzero(image > clipping_range[1])

    return {
        'n': total_points,
        'range': (min_val, max_val),
        'holes': holes,
        'mean': mean,
        'std': std,
        'clipped': (clip_min, clip_max)
    }

def transform(image: np.ndarray, center: Optional[tuple] = None, zoom: float = 1) -> np.ndarray:
    """Tranform image while preserving original dimensions

    Parameters
    ----------
    image : ndarray
        Depth or RGB image
    
    center : tuple
        New target central pixel
    
    zoom:
        New zoom/scale

    Returns
    -------
    ndarray
        New tranformed deth image
    """

    if zoom <= 1:
        return np.array(image)

    h, w = image.shape[0], image.shape[1]
    if center == None:
        center = (w//2, h//2)
    image_resized = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    h_new, w_new = image_resized.shape[0], image_resized.shape[1]

    h_min = center[1]/h * h_new - h/2
    h_min = min(max(int(h_min), 0), h_new - h)
    h_max = h_min + h
    
    w_min = center[0]/w * w_new - w/2
    w_min = min(max(int(w_min), 0), w_new - w)
    
    w_max = w_min + w
    image_fitted = image_resized[h_min:h_max, w_min:w_max,...]

    return image_fitted


def colorize(image: np.ndarray, val_range: tuple, mark_clip: bool = True, mark_holes: bool = False) -> np.ndarray:
    """Convert depth image to 3-channel format, rescale values in range, and colorize holes and clipped areas

    Parameters
    ----------
    image : ndarray
        Depth image
    
    val_range : tuple
        Target range of depth values to focus on (defines clipping range)

    mark_clip : bool
        Whether to colorize clipped pixels

    mark_holes : bool
        Whether to colorize depth holes

    Returns
    -------
    ndarray
        New colorized image
    """

    if (np.issubdtype(image.dtype, np.floating)):
        type_max = 1
    else:
        type_max = np.iinfo(image.dtype).max

    image_colored = np.zeros(image.shape + (3,), dtype=image.dtype)
    image_colored[:,:,0] = image
    image_colored[:,:,1] = image
    image_colored[:,:,2] = image

    minVal = val_range[0]
    maxVal = val_range[1]
    image_colored[image < minVal, :] = (minVal, minVal, minVal)
    image_colored[image > maxVal, :] = (maxVal, maxVal, maxVal)
    image_colored = (image_colored - minVal) * (type_max / (maxVal - minVal))
    image_colored = image_colored.astype(image.dtype)

    if mark_clip:
        image_colored[image>val_range[1], :] = (type_max,0,0)
        image_colored[(0<image) & (image<val_range[0]), :] = (0,0,type_max)
        #image2[0<image2[:,:,0]<val_range[0], :] = (60000,0,0)
    if mark_holes:
        image_colored[image==0, :] = (type_max,0,type_max)
    
    return image_colored


def denormalize_frame(frame: Union[tf.Tensor, np.ndarray], hole_bias: int = 0) -> np.ndarray:
    """Reverts tensor or ndarray back to regular uint16 depth image format

    Parameters
    ----------
    frame : Tensor | ndarray
        Normalized image
    
    hole_bias : int
        Special 0-clipping threshold

    Returns
    -------
    ndarray
        Denormalized image
    """
    if tf.is_tensor(frame):
        frame = frame.numpy()
    if len(frame.shape) == 4:
        frame = frame[0,:,:,0]
    elif len(frame.shape) == 3:
        frame = frame[:,:,0]
    frame = np.clip(frame, 0, 1)
    frame = (frame * np.iinfo('uint16').max).astype('uint16')
    frame[frame < hole_bias] = 0
    return frame