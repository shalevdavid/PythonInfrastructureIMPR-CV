# IMPR 2017, IDC
# ex4 driver

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


import ex4

# to get fixed set of random numbers
np.random.seed(seed=0)

   
def test_2ab():
    
    x_n = np.asarray([5,7,-3,2,5,9,-4,2])
    F_n = ex4.Fourier1D (x_n)
    print (F_n)
    
    x_n_rev = ex4.invFourier1D (F_n)
    print (x_n_rev)
    
    # compare results to np.fft.fft
    F_n_np = np.fft.fft(x_n)
    x_n_np_rev = np.fft.ifft(F_n_np)

    print (np.abs(F_n_np-F_n))
    print (np.abs(x_n_rev-x_n_np_rev))



def test_2cd():
    
    x_n = np.asarray([5,7,-3,2,5,9,-4,2])
    F_n_polar = ex4.Fourier1DPolar (x_n)
    print (F_n_polar)
    
    x_n_rev = ex4.invFourier1DPolar (F_n_polar)
    print (x_n_rev)

def test_3 (imageName,upsamplingFactor):
    
    
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    
    upsampledImage, Fimg, zeroPaddedFimg = ex4.imageUpsampling (img, upsamplingFactor)
    f, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Original')
    ax2.imshow(upsampledImage, cmap='gray',vmin=0, vmax=255), ax2.set_title('Upsampled (' + str(upsamplingFactor[0]) +', ' + str(upsamplingFactor[1]) +')')
    ax3.imshow(np.log(1 + np.abs(Fimg)), cmap='gray'), ax3.set_title('FFT')
    ax4.imshow(np.log(1 + np.abs(zeroPaddedFimg)), cmap='gray'), ax4.set_title('Zero padded FFT')
        
                       
def test_4 (dx,dy):
    
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    r,c = img.shape


    img1 = img[0:r-dy,0:c-dx]
    img2 = img[dy:r,dx:c]
    
    res_dx, res_dy, phaseCorr = ex4.phaseCorr (img1, img2)

   
    f, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex='col')
    ax1.imshow(img1, cmap='gray',vmin=0, vmax=255), ax1.set_title('Fixed image')
    ax2.imshow(img2, cmap='gray',vmin=0, vmax=255), ax2.set_title('Moving image (dx=' + str(dx) + ', dy=' + str(dy) + ')')
    ax3.imshow(np.abs(phaseCorr), cmap='gray'), ax3.set_title('PhaseCorr (dx=' + str(res_dx) + ', dy=' +str(res_dy) +')')
        

def test_5 (imageName, lowThresh,highThresh):
    
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    filtImage, Fimg, mask = ex4.imFreqFilter (img, lowThresh, highThresh)   
    
    f, ((ax1, ax2), (ax3,ax4))  = plt.subplots(2, 2, sharex='col', sharey='row')
    plt.suptitle('low = ' + str(lowThresh) + ' high = ' + str(highThresh))
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Image')
    ax2.imshow(np.log(np.abs(Fimg)+1), cmap='gray'), ax2.set_title('FFT')
    ax3.imshow(mask, cmap='gray', vmin=0, vmax=1), ax3.set_title('Mask')
    ax4.imshow(filtImage, cmap='gray', vmin=0, vmax=255), ax4.set_title('Filtered image')


def test_5d1():
    
    # create motion degraded image 
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    r,c = img.shape
    size = 55

    # generating motion kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / np.sum(kernel_motion_blur.flatten())
    imgMotion=convolve2d(img, kernel_motion_blur, mode='same', boundary='wrap')
    
    
    k = 0.000000000001 # no added noise
    recImg = ex4.imageDeconv (imgMotion, kernel_motion_blur, k)
#    
    f, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex='col')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Image')
    ax2.imshow(imgMotion, cmap='gray'), ax2.set_title('Degraded image')
    ax3.imshow(recImg, cmap='gray'), ax3.set_title('recovered from degraded image')
    
def test_5d2():
    
    # create motion degraded image 
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    r,c = img.shape
    size = 25

    # generating echo kernel
    kernel_echo_blur = np.zeros((size, size))
    kernel_echo_blur[12,12] = 1
    kernel_echo_blur[12,22:25] = 1
    kernel_echo_blur[22:25,12] = 1
    kernel_echo_blur = kernel_echo_blur / np.sum(kernel_echo_blur.flatten())
    imgEcho=convolve2d(img, kernel_echo_blur, mode='same', boundary='wrap')
    
    
    k = 0.000000000001 # no added noise
    recImg = ex4.imageDeconv (imgEcho, kernel_echo_blur, k)
#    
    f, (ax1, ax2, ax3)  = plt.subplots(1, 3, sharex='col')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Image')
    ax2.imshow(imgEcho, cmap='gray'), ax2.set_title('Degraded image')
    ax3.imshow(recImg, cmap='gray'), ax3.set_title('recovered from degraded image')
    
    
if __name__ == "__main__":

    plt.show()
    plt.ion()

#    # test 2 - 1D Fourier implementation
    test_2ab()
    test_2cd()

    #test 3 - image upsampling
    imageName = './Images/cameraman.tif'
    test_3(imageName, [2,2])
    test_3(imageName, [3,2])
    test_3(imageName, [2,4])
    test_3(imageName, [0.5,0.5])
    
    # test 4 - phase correlation
    test_4(10,5)
    test_4(33, 27)
    test_4(76, 100)
    
    # test 5
    
    # low pass filtering
    lowThresh = 0
    highThresh = 5
    imageName = './Images/cameraman.tif'
    test_5 (imageName, lowThresh,highThresh)

    lowThresh = 0
    highThresh = 15
    imageName = './Images/cameraman.tif'
    test_5(imageName, lowThresh, highThresh)

    lowThresh = 0
    highThresh = 30
    imageName = './Images/cameraman.tif'
    test_5(imageName, lowThresh, highThresh)

    lowThresh = 0
    highThresh = 400
    imageName = './Images/cameraman.tif'
    test_5(imageName, lowThresh, highThresh)

    # high pass filtering
    lowThresh = 1
    highThresh = 400
    imageName = './Images/cameraman.tif'
    test_5 (imageName, lowThresh,highThresh)

    lowThresh = 3
    highThresh = 400
    imageName = './Images/cameraman.tif'
    test_5(imageName, lowThresh, highThresh)

    lowThresh = 10
    highThresh = 400
    imageName = './Images/cameraman.tif'
    test_5(imageName, lowThresh, highThresh)

    lowThresh = 100
    highThresh = 400
    imageName = './Images/cameraman.tif'
    test_5(imageName, lowThresh, highThresh)

    # band pass filtering
    lowThresh = 5
    highThresh = 15
    imageName = './Images/cameraman.tif'
    test_5 (imageName, lowThresh,highThresh)

    lowThresh = 10
    highThresh = 15
    imageName = './Images/cameraman.tif'
    test_5 (imageName, lowThresh,highThresh)

    lowThresh = 15
    highThresh = 20
    imageName = './Images/cameraman.tif'
    test_5 (imageName, lowThresh,highThresh)

    lowThresh = 20
    highThresh = 40
    imageName = './Images/cameraman.tif'
    test_5 (imageName, lowThresh,highThresh)

    # deconvolution
    test_5d1()
    
    test_5d2()

    dummy = 0
