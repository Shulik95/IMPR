import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

MAX_GRAY_SCALE = 255


def read_image(file_name, representation):
    """
    reads an image file and converts it into given representation.
    :param file_name: the filename of an image on disk.
    :param representation: representation code, either 1 or 2 defining whether
    should be a grayscale image (1) or an RGB image (2).
    :return: an image with intensities normalized to the range [0,1]
    """
    img = np.array(imread(file_name))
    img_float = img.astype(np.float64)
    if representation == 1:  # return grayscale image
        if img.ndim == 2:  # image was given in grayscale
            return img_float
        elif img.ndim == 3:  # image is rgb, convert to grayscale
            return rgb2gray(img_float)
    elif representation == 2:  # return rgb
        return img_float


def imdisplay(filename, representation):
    """
    utilizes read_image to display an image in given representation
    :param filename: same as is read_image
    :param representation: same as in read_image
    """
    plt.imshow(read_image(filename, representation))
    plt.show()


if __name__ == '__main__':
    plt.imshow(read_image("graygoose.jpg", 1), cmap="gray")
    plt.show()
    pass