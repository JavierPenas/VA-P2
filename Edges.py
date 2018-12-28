import cv2
import scipy.ndimage as nd
import Loader as ld

def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F);


def laplacian_of_gaussian(img, sig):
    return nd.gaussian_laplace(img, sigma=sig)


def difference_of_gaussian(img,sigma1, sigma2):
    # run a 5x5 gaussian blur then a 3x3 gaussian blr
    s1 = cv2.GaussianBlur(img, (11, 11), sigma1)
    s2 = cv2.GaussianBlur(img, (11, 11), sigma2)

    return s2 - s1


def canny(img, th1, th2, sigma: float = 0):
    if sigma != 0:
        img = cv2.GaussianBlur(img, (11, 11), sigma)
        ld.print_image(img)
    return cv2.Canny(img, th1, th2)
