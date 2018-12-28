import Loader
import cv2
import numpy as np
from matplotlib import pyplot as plt
import Smoothing as smooth
import Thresholding as thresh
from skimage.restoration import denoise_tv_chambolle
import Edges as edg
from skimage import exposure

def original_test_cases():
    img1 = Loader.load_image("im1.jpeg")
    img2 = Loader.load_image("im2.jpeg")
    # img3 = Loader.load_image("im3.jpeg")
    # img4 = Loader.load_image("im4.jpeg")
    Loader.hist_compare([img1, img2], ["Example 1", "Example 2"], True)
    Loader.hist_and_cumsum(img1)



def denoising_comparison(img, hist: bool = False):
    # NON LOCAL MEAN DENOISING
    nl_means_denoised_img = smooth.denoising_NlMeans(img)
    # MEDIAN FILTER DENOISING
    mean_denoised_img = cv2.medianBlur(img, 9)
    mean_denoised_img = cv2.medianBlur(mean_denoised_img, 9)
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


def test1(denoised_imgs, denoised_titles):

    output = denoised_imgs.copy()
    for i in np.arange(0, len(denoised_imgs)):
        # output[i] = thresh.apply_thresholding_algorithm(denoised_imgs[i], 2)
        # Contrast stretching
        p2, p98 = np.percentile(denoised_imgs[i], (2, 98))
        output[i] = exposure.rescale_intensity(denoised_imgs[i], in_range=(p2, p98))

    for i in np.arange(1, len(denoised_imgs)):
        Loader.hist_compare([denoised_imgs[0], output[i]],
                            ["Original", denoised_titles[i]], True)

    for i in np.arange(0, len(denoised_imgs)):
        output[i] = thresh.apply_thresholding_algorithm(output[i], 2)

    for i in np.arange(1, len(denoised_imgs)):
        Loader.hist_compare([denoised_imgs[0], output[i]],
                            ["Original", denoised_titles[i]], False)


if __name__ == '__main__':
    img = Loader.load_image("im3.jpeg")

    # print("[DEBUG] Printing original histograms for 4 examples")
    #original_test_cases()
    # denoising_comparison(img)
    #TOTAL VARIATION DENOISE
    #imgs, titles = denoising_comparison(img, True)
    #test1(imgs, titles)
    # MEDIAN FILTER DENOISING
    mean_denoised_img = cv2.medianBlur(img, 9)
    mean_denoised_img = cv2.medianBlur(mean_denoised_img, 9)
    #edges = edg.laplacian_of_gaussian(mean_denoised_img, 2)
    edges = edg.difference_of_gaussian(mean_denoised_img, 1.0, 1.5)
    test = cv2.medianBlur(edges, 7)
    #edges = edg.canny(mean_denoised_img, 100, 70, sigma=0.0001)
    Loader.print_image(test)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    opening = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)
    Loader.hist_compare([test, opening], ["test", "opening"])
    print("[DEBUG] End of processing")
