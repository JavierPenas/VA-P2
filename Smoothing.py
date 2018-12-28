import cv2
import scipy.ndimage.filters as filters


def denoising_NlMeans(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def gaussian(img, sigma: float):
    return filters.gaussian_filter(img, sigma)


def min_filter(img, size):
    return filters.minimum_filter(img, size)


def max_filter(img, size):
    return filters.maximum_filter(img, size)


def sobel(image, axis=-1):
    return filters.sobel(image, axis)


def laplacian(img, sigma=None):

    if sigma is not None :
        return filters.gaussian_laplace(img, sigma)
    else:
        return filters.laplace(img)
