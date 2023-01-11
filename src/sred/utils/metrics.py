from skimage.metrics import structural_similarity as ssim
import numpy as np


def nmid(original: np.ndarray, denoised: np.ndarray, max_val: int = np.iinfo('uint16').max):
    """Non-reference metric for measuring image denoising

    Parameters
    ----------
    original : np.ndarray
        Original image
    
    denoised : np.ndarray
        Denoised image

    max_val : int
        Maximum possible pixel value
    
    Returns
    -------
    float
        NMID value
    """

    #if np.all(original == denoised):
    #    return 1.0
    
    original = original.astype('float64')
    denoised = denoised.astype('float64')
    
    MNI = original - denoised
    _, N = ssim(original, MNI, data_range=max_val, full=True)
    _, P = ssim(original, denoised, data_range=max_val, full=True)
    e = np.corrcoef(N.flatten(), P.flatten())
    return e[0,1]


def temporal_diff(img1: np.ndarray, img2: np.ndarray) -> tuple[float]:
    """Non-reference temporal coherence metric

    Parameters
    ----------
    img1 : np.ndarray
        First image
    
    img2 : np.ndarray
        Second image
    
    Returns
    -------
    (float, float)
        Mean and vsriance of the pixels
    """

    diff = img2.astype('float64') - img1.astype('float64')
    return np.mean(diff), np.var(diff)