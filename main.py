import Loader
import cv2
import numpy as np
from matplotlib import pyplot as plt
import Smoothing as smooth
import Thresholding as thresh
from skimage.restoration import denoise_tv_chambolle
import Edges as edg
from skimage import exposure


#
# COMPARTIVA DE MÉTODOS PROBADOS PARA ELIMINAR
# EL RUIDO MOTEADO PRESENTE EN LAS IMAGENES DE EJEMPLO
#
def denoising_comparison(img, hist: bool = False):
    # NON LOCAL MEAN DENOISING
    nl_means_denoised_img = smooth.denoising_NlMeans(img)
    # MEDIAN FILTER DENOISING
    mean_denoised_img = smooth.median_filter(img, 9)
    mean_denoised_img = smooth.median_filter(img, 9)
    # GAUSSIAN DENOISING
    gaussian_denoised = smooth.gaussian(img, 1.5)
    # MINIMUM FILTER
    minimum_denoised = smooth.min_filter(img, (5, 5))
    # MAXIMUM FILTER
    maximum_denoised = smooth.max_filter(img, (5, 5))

    denoised_imgs = [img, gaussian_denoised, mean_denoised_img, nl_means_denoised_img, minimum_denoised, maximum_denoised]
    denoised_titles = ["Original", "Denoised Gaussian", "Median Filtered", "NL Means Filter", "Minimums Filter", "Maximums Filter"]
    Loader.hist_compare(denoised_imgs, denoised_titles, hist)

    return denoised_imgs, denoised_titles


#
# COMPARTIVA DE MÉTODOS PROBADOS PARA
# REALIZAR LA SEGMENTACION DE LA IMAGEN
#
def thresholding_comparison(img):

    #Thresholding algoritms precalculation for comparison
    triangle = thresh.apply_thresholding_algorithm(img, thresh.THRESH_TRIANGLE)
    mean = thresh.apply_thresholding_algorithm(img, thresh.THRESH_MEAN)
    otsu = thresh.apply_thresholding_algorithm(img, thresh.THRESH_OTSU)
    yen = thresh.apply_thresholding_algorithm(img, thresh.THRESH_YEN)
    minimum = thresh.apply_thresholding_algorithm(img, thresh.THRESH_MIMIMUM)
    isodata = thresh.apply_thresholding_algorithm(img, thresh.THRESH_ISODATA)
    # li = thresh.apply_thresholding_algorithm(img, thresh.THRESH_LI)

    thresholded_imgs = [img, triangle, mean, otsu, yen,
                        minimum, isodata]
    thresholded_titles = ["Original", "Triangle", "Mean",
                          "Otsu", "Yen", "Minimum", "Isodata"]
    Loader.hist_compare(thresholded_imgs, thresholded_titles)


def edgesFunctions(img):
    log = edg.laplacian_of_gaussian(img, 2)
    dog = edg.difference_of_gaussian(img, 1.0, 1.5)
    canny = edg.canny(img, 100, 200, sigma=1.5)
    Loader.hist_compare([log, dog, canny], ["LoG", "DoG", "Canny"])


if __name__ == '__main__':

    print("[DEBUG] Load image from local sources")
    img = Loader.load_image("im1.jpeg")
    # print("[DEBUG] Showing VISUAL denoising algorithm comparison")
    # denoising_comparison(img)
    # print("[DEBUG] Showing HISTOGRAM denoising algorithm comparison")
    # denoising_comparison(img, True)

    # DENOISING IMAGE
    denoised_img = smooth.median_filter(img, 9)
    denoised_img = smooth.median_filter(denoised_img, 9)
    # PRINT DENOISED IMAGE AND HISTOGRAM
    # Loader.print_image(denoised_img)
    # Loader.hist_and_cumsum(denoised_img)

    ##TRIAL ZONE --- MAY BE ALL IS WRONG! --

    # thresholding_comparison(denoised_img)
    th_img = thresh.apply_thresholding_algorithm(denoised_img, thresh.THRESH_MEAN)
    back, front = thresh.get_regions(denoised_img, th_img)
    # Loader.hist_compare([back, front], ["Back", "Front"])
    # edges = edg.laplacian_of_gaussian(front, 2)
    Loader.print_image(front)
    eq = Loader.equalization(front.astype("uint8"))
    Loader.print_image(eq)
    Loader.hist_and_cumsum(eq)
    # edges = edg.laplacian_of_gaussian(eq, 2)
    # Loader.print_image(edges)
    # edgesFunctions(eq)

    # EDGE DETECTION

    ## END OF TRIAL ZONE -- YOU CAN TAKE OF THE HELMET --

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    # opening = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)
    # Loader.hist_compare([test, opening], ["test", "opening"])
    print("[DEBUG] End of processing")
