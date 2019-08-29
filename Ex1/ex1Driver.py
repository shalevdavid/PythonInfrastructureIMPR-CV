# IMPR 2017, IDC
# ex1 driver

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ex1

# to get fixed set of random numbers
np.random.seed(seed=0)


def test_1():
    
    dim=[0,1,0,1]; # image spatial extent
    
    f, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
    
    PixelSize=1.0/500
    I1 = ex1.getSampledImageAtResolution (dim, PixelSize,2)
    ax1.imshow(I1, cmap='gray',extent=(dim[0],dim[1],dim[2],dim[3])), ax1.set_xlim([0,1]), ax1.set_ylim([0,1]), ax1.autoscale(False), ax1.set_title('PixelSize = ' + str (PixelSize))
    
    PixelSize=1.0/250
    I2 = ex1.getSampledImageAtResolution (dim, PixelSize,2)
    ax2.imshow(I2, cmap='gray',extent=(dim[0],dim[1],dim[2],dim[3])), ax2.set_xlim([dim[0],dim[1]]), ax2.set_ylim([dim[2],dim[3]]), ax2.autoscale(False), ax2.set_title('PixelSize = ' + str (PixelSize))
    
    PixelSize=1.0/100
    I3 = ex1.getSampledImageAtResolution (dim, PixelSize,2)
    ax3.imshow(I3, cmap='gray',extent=(dim[0],dim[1],dim[2],dim[3])), ax3.set_xlim([dim[0],dim[1]]), ax3.set_ylim([dim[2],dim[3]]), ax3.autoscale(False), ax3.set_title('PixelSize = ' + str (PixelSize))
    
    PixelSize=1.0/50
    I4 = ex1.getSampledImageAtResolution (dim, PixelSize,2)
    ax4.imshow(I4, cmap='gray',extent=(dim[0],dim[1],dim[2],dim[3])), ax4.set_xlim([dim[0],dim[1]]), ax4.set_ylim([dim[2],dim[3]]), ax4.autoscale(False), ax4.set_title('PixelSize = ' + str (PixelSize))
    
    PixelSize=1.0/25
    I5 = ex1.getSampledImageAtResolution (dim, PixelSize,2)
    ax5.imshow(I5, cmap='gray',extent=(dim[0],dim[1],dim[2],dim[3])), ax5.set_xlim([dim[0],dim[1]]), ax5.set_ylim([dim[2],dim[3]]), ax5.autoscale(False), ax5.set_title('PixelSize = ' + str (PixelSize))
    
    PixelSize=1.0/10
    I6 = ex1.getSampledImageAtResolution (dim, PixelSize,2)
    ax6.imshow(I6, cmap='gray',extent=(dim[0],dim[1],dim[2],dim[3])), ax6.set_xlim([dim[0],dim[1]]), ax6.set_ylim([dim[2],dim[3]]), ax6.autoscale(False), ax6.set_title('PixelSize = ' + str (PixelSize))
    
def test_2(imageName):
    img = cv2.imread(imageName)
    
    qImg = ex1.optimalQuantizationImage(img,64) 
    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')
    
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(qImg, cmap='gray'), ax2.set_title('Quantized: 64')
    
    
    qImg = ex1.optimalQuantizationImage(img,16) 
    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')
    
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(qImg, cmap='gray'), ax2.set_title('Quantized: 16')
    
    qImg = ex1.optimalQuantizationImage(img,4) 
    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')
    
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(qImg, cmap='gray'), ax2.set_title('Quantized: 4')
    

    qImg = ex1.optimalQuantizationImage(img,2) 
    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')
    
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(qImg, cmap='gray'), ax2.set_title('Quantized: 2')
    
def test_3_a(imageName):
    
    img = cv2.imread(imageName)
    hist = ex1.getImageHistogram(img)
    x = np.arange(0,256,1)
    plt.figure()
    plt.bar(x, hist, color="blue")
 
def test_3_b(imageName):

    img = cv2.imread(imageName)
    hist = ex1.getImageHistogram(img)
    img_c = ex1.getConstrastStrechedImage(img)
    hist_c = ex1.getImageHistogram(img_c)
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(img_c, cmap='gray'), ax2.set_title('Contrast enhanced')
    
    x = np.arange(0,256,1)
    ax3.bar(x, hist, color="blue"), ax3.set_xlim([0, 255])
    ax4.bar(x, hist_c, color="blue"), ax4.set_xlim([0, 255])
 
def test_3_c(imageName):
        
    img = cv2.imread(imageName)
    histEqRes= ex1.getHistEqImage(img)
     
    hist = ex1.getImageHistogram(img)
    hist_eq = ex1.getImageHistogram(histEqRes)
    
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
    
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(histEqRes, cmap='gray'), ax2.set_title('Histogram equalizaed')
    
    x = np.arange(0,256,1)
    ax3.bar(x, hist, color="blue"), ax3.set_xlim([0, 255]),ax3.autoscale(False)
    ax4.bar(x, hist_eq, color="blue"), ax4.set_xlim([0, 255]),ax4.autoscale(False)
 
    ax5.plot(x, np.cumsum(hist), color="blue"), ax5.set_xlim([0, 255]),ax5.autoscale(False)
    ax6.plot(x, np.cumsum(hist_eq), color="blue"), ax6.set_xlim([0, 255]),ax6.autoscale(False)
 

if __name__ == "__main__":
    
    # test 1.
    test_1()

    # test 2.
    imageName = './Images/cameraman.jpg'
    test_2(imageName)
    
#    # test 3.a.
    imageName = './Images/cameraman.jpg'
    test_3_a(imageName)
#    
#    # test 3.b.
    imageName = './Images/im1.png'
    test_3_b(imageName)
#    
#    # test 3.c.
    imageName = './Images/im2.png'
    test_3_c(imageName)
#    
#    
    dummy = 0