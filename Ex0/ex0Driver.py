# IMPR 2017, IDC
# ex0 driver

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ex0

# to get fixed set of random numbers
np.random.seed(seed=0)


def test_1_a():
    Z, Zmin, ZminRow, ZminCol, Zmax, ZmaxRow, ZmaxCol = ex0.retrunRandomMatrixWithMinMax(13)
    print(Z)
    print(Zmin, ZminRow, ZminCol)
    print(Zmax, ZmaxRow, ZmaxCol)


def test_1_b():
    coords = np.random.random((5, 2))
    print(coords)
    p_coords = ex0.cartesian2polar2D(coords)
    print(p_coords)


def test_2(imageName):
    img = cv2.imread(imageName);

    # reorder from BGR to RGB
    img = img[:, :, [2, 1, 0]]

    g_img_Lightness = ex0.convertRGB2Gray(img, 'Lightness')
    g_img_Average = ex0.convertRGB2Gray(img, 'Average')
    g_img_Luminosity = ex0.convertRGB2Gray(img)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(img), ax1.set_title('RGB image')
    ax2.imshow(g_img_Lightness, cmap='gray'), ax2.set_title('Grayscale image: Lightness')
    ax3.imshow(g_img_Average, cmap='gray'), ax3.set_title('Grayscale image: Average')
    ax4.imshow(g_img_Luminosity, cmap='gray'), ax4.set_title('Grayscale image: Luminosity')


def test_3(imageName):
    img = cv2.imread(imageName);

    # reorder from BGR to RGB
    img = img[:, :, [2, 1, 0]]

    # normalize to range [0 1]
    img = np.float32(img)
    img = img / 255

    img_yiq = ex0.rgb2yiq(img)
    img_rgb = ex0.yiq2rgb(img_yiq)

    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col')
    ax1.imshow(img), ax1.set_title('Original RGB image')
    ax2.imshow(img_rgb), ax2.set_title('rgb2yiq->yiq2rgb image')


if __name__ == "__main__":
    # test 1.a.
    test_1_a()

    # test 1.b.
    test_1_b()

    # test 2.
    imageName = './Images/im1.jpg'
    test_2(imageName)

    imageName = './Images/peppers.png'
    test_2(imageName)

    # test 3.
    imageName = './Images/im1.jpg'
    test_3(imageName)

    imageName = './Images/peppers.png'
    test_3(imageName)
