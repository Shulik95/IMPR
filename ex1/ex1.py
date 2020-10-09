# ---------- imports ---------- #
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# ---------- const definitions ---------- #
MAX_GRAY_SCALE = 255
TWO_DIM = 2
THREE_DIM = 3

# transformation matrix for rgb to yiq #
rgb2yiq_matrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])


# ---------- functions ---------- #


def read_image(file_name, representation):
    """
    reads an image file and converts it into given representation.
    :param file_name: the filename of an image on disk.
    :param representation: representation code, either 1 or 2 defining whether
    should be a grayscale image (1) or an RGB image (2).
    :return: an image with intensities normalized to the range [0,1]
    """
    img = np.array(imread(file_name))
    img_float = img.astype(np.float64) / MAX_GRAY_SCALE
    if representation == 1:  # return grayscale image
        if img.ndim == TWO_DIM:  # image was given in grayscale
            return img_float
        elif img.ndim == THREE_DIM:  # image is rgb, convert to grayscale
            return rgb2gray(img_float)
    elif representation == 2:  # return rgb
        return img_float


def imdisplay(filename, representation):
    """
    utilizes read_image to display an image in given representation
    :param filename: same as is read_image
    :param representation: same as in read_image
    """
    plt.imshow(read_image(filename, representation), cmap='gray')
    plt.show()


def rgb2yiq(imRGB):
    """
    transforms an RGB image into the YIQ color space.
    :param imRGB: image to transform.
    :return: the YIQ values as a matrix
    """
    ret_arr = np.zeros(shape=imRGB.shape, dtype=np.float64)  # same size matrix
    for i in range(THREE_DIM):
        ret_arr[:, :, i] = rgb2yiq_matrix[i][0] * imRGB[:, :, 0] +\
                           rgb2yiq_matrix[i][1] * imRGB[:, :, 1] +\
                           rgb2yiq_matrix[i][2] * imRGB[:, :, 2]
    return ret_arr


def yiq2rgb(imYIQ):
    """
    transforms an YIQ image into the RGB color space.
    :param imYIQ: image to transform.
    :return:
    """
    

if __name__ == '__main__':
    print(rgb2yiq(read_image("rgbimage.jpg", 2)).ndim)
