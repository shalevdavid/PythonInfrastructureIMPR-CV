# IMPR 2017, IDC
# ex5 driver

import ex5
import matplotlib.pyplot as plt
import cv2
import numpy as np

import timeit  # for running time tests

def test_1():
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)


    numOfLevels = 6
    g_pyr = ex5.gaussianPyramid (img, numOfLevels, filterParam=0.4)
    
    f, (ax1, ax2, ax3,ax4, ax5, ax6)  = plt.subplots(1, 6, sharex='col')
    ax1.imshow(g_pyr[0], cmap='gray'), ax1.set_title('G-0')
    ax2.imshow(g_pyr[1], cmap='gray'), ax2.set_title('G-1')
    ax3.imshow(g_pyr[2], cmap='gray'), ax3.set_title('G-2')
    ax4.imshow(g_pyr[3], cmap='gray'), ax4.set_title('G-3')
    ax5.imshow(g_pyr[4], cmap='gray'), ax5.set_title('G-4')
    ax6.imshow(g_pyr[5], cmap='gray'), ax6.set_title('G-5')

def test_1_bonus2 ():
    try:
    
        runTime = timeit.timeit(stmt='ex5.imConv2(img,kernel1D)', \
                                setup = 'import cv2; import ex5; imageName = "./Images/cameraman.tif"; img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE); kernel1D = ex5.getKernel(0.4)',
                                number=100)
        
        print ("Convolution operator running time for 100 runs: " + str (runTime))

        runTime = timeit.timeit(stmt='ex5.convolution2D(img,kernel1D[:,None]*kernel1D[None,:])', \
                                setup = 'import cv2; import ex5; imageName = "./Images/cameraman.tif"; img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE); kernel1D = ex5.getKernel(0.4)',
                                number=100)

        print("school 2D Convolution operator using fft2D only running time for 100 runs: " + str(runTime))

        runTime = timeit.timeit(stmt='scisig.convolve2d(img,kernel1D[:,None]*kernel1D[None,:],\'same\')', \
                                setup='import cv2; import scipy.signal as scisig; import ex5; imageName = "./Images/cameraman.tif"; img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE); kernel1D = ex5.getKernel(0.4)',
                                number=100)

        print("scipy.signal 2D Convolution operator running time for 100 runs: " + str(runTime))

    except:
    
        print ("Bonus 2 is not implemented")
        
def test_2ab():
    
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)

    numOfLevels = 6
    
    l_pyr = ex5.laplacianPyramid (img, numOfLevels, filterParam=0.4)
    f, (ax1, ax2, ax3,ax4, ax5, ax6)  = plt.subplots(1, 6, sharex='col')
    ax1.imshow(l_pyr[0], cmap='gray'), ax1.set_title('L-0')
    ax2.imshow(l_pyr[1], cmap='gray'), ax2.set_title('L-1')
    ax3.imshow(l_pyr[2], cmap='gray'), ax3.set_title('L-2')
    ax4.imshow(l_pyr[3], cmap='gray'), ax4.set_title('L-3')
    ax5.imshow(l_pyr[4], cmap='gray'), ax5.set_title('L-4')
    ax6.imshow(l_pyr[5], cmap='gray'), ax6.set_title('L-5')


    l_pyr_recon = ex5.imgFromLaplacianPyramid(l_pyr, numOfLevels, filterParam=0.4)
    
    f, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex='col')
    ax1.imshow(img, cmap='gray'), ax1.set_title('Org')
    ax2.imshow(l_pyr_recon, cmap='gray'), ax2.set_title('Laplacian Pyr recon')
    ax3.imshow(l_pyr_recon-img, cmap='gray'), ax3.set_title('Diff')


def test_3():
    image1Name = './Images/black.tif'
    img1 = cv2.imread(image1Name,cv2.IMREAD_GRAYSCALE)
    img1=img1[0:512,200:712]
    
    image2Name = './Images/white.tif'
    img2 = cv2.imread(image2Name,cv2.IMREAD_GRAYSCALE)
    img2=img2[0:512,200:712]
    
    blendingMaskName = './Images/mask.png'
    blendingMask = cv2.imread(blendingMaskName,cv2.IMREAD_GRAYSCALE)
    blendingMask = np.clip(blendingMask,0,1)
    blendingMask = blendingMask[0:512,200:712]
    

    numOfLevels = 6
    blendedImg = ex5.imgBlending(img1,img2,blendingMask, numOfLevels, filterParam=0.4)
    f, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex='col')
    ax1.imshow(img1, cmap='gray'), ax1.set_title('Img1')
    ax2.imshow(img2, cmap='gray'), ax2.set_title('Img2')
    ax3.imshow(blendedImg, cmap='gray'), ax3.set_title('Blended')
    
    
if __name__ == "__main__":

    plt.ion()
    plt.show()

    # test 1 - Gaussain pyramid
    test_1()
    
    # test 1 bonus 2
    test_1_bonus2 ()
    
    # test 2 - Laplacian pyramid forward - backward
    test_2ab()
    
    
    # test 3 - Image blending
    test_3()

    dummy = 0
    