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
    dog = edg.difference_of_gaussian(img, 1.0, 2.5)
    sobelX = edg.sobel(img, 0)
    canny = edg.canny(img, 100, 200, sigma=1.5)
    Loader.hist_compare([log, dog, canny, sobelX], ["LoG", "DoG", "Canny", "Sobel"])


def fill_cornea(edges):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 15))
    stretched = cv2.morphologyEx(closed, cv2.MORPH_ERODE, kernel)

    #Loader.hist_compare([edges,dilated,closed,stretched],["Original","Dilated","Closed","Stretched"])
    #cv2.imwrite(Loader.BASE_PATH + "res/edgesIMG.jpeg", img)
    return stretched


def process_image(img):
    # Loader.print_image(img)
    # print("[DEBUG] Showing VISUAL denoising algorithm comparison")
    # denoising_comparison(img)
    # print("[DEBUG] Showing HISTOGRAM denoising algorithm comparison")
    # denoising_comparison(img, True)

    # DENOISING IMAGE
    denoised_img = smooth.median_filter(img, 9)
    denoised_img = smooth.median_filter(denoised_img, 7)
    # PRINT DENOISED IMAGE AND HISTOGRAM
    #Loader.print_image(denoised_img)
    # Loader.hist_and_cumsum(denoised_img)

    # thresholding_comparison(denoised_img)
    th_img = thresh.apply_thresholding_algorithm(denoised_img, thresh.THRESH_TRIANGLE)
    back, front = thresh.get_regions(denoised_img, th_img)
    # Loader.hist_compare([back, front], ["Back", "Front"])
    #Loader.print_image(front)
    eq = Loader.equalization(front.astype("uint8"))
    #Loader.print_image(eq)
    # Loader.hist_and_cumsum(eq)

    # EDGE DETECTION
    #edgesFunctions(eq) # Comparison of different edge detection method
    edges = edg.laplacian_of_gaussian(eq, 2)
    #Loader.print_image(edges)
    # Fill the cornea area with white pixels
    dilated = fill_cornea(edges)
    # Surround the córnea area and lens edges with visible and thin line
    (i, contornos, jerarquia) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contornos, -1, (0, 0, 255), 3)
    #Loader.print_image(img)
    return img
    # cv2.imwrite(Loader.BASE_PATH+"res/img10.jpeg", img)


def apply_to_all():
    for i in np.arange(1, 13):

        img = Loader.load_image("im"+i.astype(str)+".jpeg")
        #TODO process image must return the output image
        process_image(img)
        print("res/img"+i.astype(str)+".jpeg")
        cv2.imwrite(Loader.BASE_PATH + "res/img"+i.astype(str)+".jpeg", img)


def bright_and_contrast(source, contrast, bright):
    new_image = cv2.convertScaleAbs(source, alpha=contrast, beta=bright)
    #Loader.print_image(new_image)
    return new_image


if __name__ == '__main__':

    print("[DEBUG] Load image from local sources")
    img = Loader.load_image("im4.jpeg")
    res = process_image(img)
    Loader.print_image(res)
    #apply_to_all()
    print("[DEBUG] End of processing")

