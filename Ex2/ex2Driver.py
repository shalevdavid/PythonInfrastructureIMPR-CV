# IMPR 2017, IDC
# ex2 driver

import numpy as np
import cv2
import matplotlib.pyplot as plt

import ex2

# to get fixed set of random numbers
np.random.seed(seed=0)
plt.show()

def test_1_a_b():
    
   
    imageName = './Images/cameraman.tif'
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
    r,c = img.shape
    pts1 = np.float32([[0, 5],[20, 30],[15, 12]])

##   test case to create pts2 by known affine matrix    
#    deg=5*np.pi/180;
#    scalex = 0.75
#    scaley = 1.25
#    tx = 10
#    ty = 15
#    rotMat=np.array([[scalex*np.cos(deg), -np.sin(deg), 0],[scaley*np.sin(deg), np.cos(deg), 0],[tx, ty, 1]]).T
#    
#    print (rotMat)
#    pts1 = np.vstack((pts1.T, np.ones(pts1.shape[0])))
#    pts2 = np.dot(rotMat,pts1) # image spatial extent
#    
#    print(pts1)
#    print(pts2)
    pts2 = np.float32([[11, 20],[28, 43],[23, 26]])
   
    
    affineT = ex2.getAffineTransformation(pts1,pts2)
    print (affineT)
    
    imgT = ex2.applyAffineTransToImage(img, affineT)
    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')
    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(imgT, cmap='gray'), ax2.set_title('Transformed')

     
 
    
def test_1_c():
    img = np.zeros((256,256),dtype=np.float32)

    p0=[20,20]
    p1=[20,120]
    p2=[30,20]
    p3=[30,120]
    p4=[70,30]
    p6=[70,20]
    p7=[30,30]
    p8=[30,60]
    p9=[60,70]
    
    img[p0[1]:p3[1],p0[0]:p3[0]]=255
    img[p2[1]:p4[1],p2[0]:p4[0]]=255
    img[p8[1]:p9[1],p8[0]:p9[0]]=255
    
    f, (ax1, ax2)  = plt.subplots(1, 2, sharex='col')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Original')
    
    Qs = np.asarray([[20,120],[20,20],[30,60]], dtype=np.float32)
    Ps = np.asarray([[20,20],[70, 20],[60,60]], dtype=np.float32)
    
    Qt = np.asarray([[20,120],[20,20],[30,60]], dtype=np.float32)
    Pt = np.asarray([[20,20],[70, 10],[60,60]], dtype=np.float32)
    
    for lineIdx in np.arange(0,Qs.shape[0]):
        ax1.plot([Qs[lineIdx,0], Ps[lineIdx,0]],[Qs[lineIdx,1],Ps[lineIdx,1]],'r')
        ax1.plot([Qt[lineIdx,0], Pt[lineIdx,0]],[Qt[lineIdx,1],Pt[lineIdx,1]],'g--')
    
    p=1.0
    b=2.0
    imgT = ex2.multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, p, b)
    ax2.imshow(imgT, cmap='gray',vmin=0, vmax=255), ax2.set_title('Transformed')
        
def test_2_c(imageName):
    
    img = cv2.imread(imageName,cv2.IMREAD_GRAYSCALE)
#    
    Gx, Gy, Gmag = ex2.imGradSobel(img)
    f, ((ax1, ax2), (ax3, ax4) ) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(img, cmap='gray',vmin=0, vmax=255), ax1.set_title('Original')
    ax2.imshow(Gx, cmap='gray',vmin=0, vmax=255), ax2.set_title('Gx')
    ax3.imshow(Gy, cmap='gray',vmin=0, vmax=255), ax3.set_title('Gy')
    ax4.imshow(Gmag, cmap='gray',vmin=0, vmax=255), ax4.set_title('Gmag')

   
    


if __name__ == "__main__":

#    # test 1.
    test_1_a_b()
#
#
    test_1_c()
#
#
    # test 2.
    imageName = './Images/cameraman.tif'
    test_2_c(imageName)
    
    imageName = './Images/clown.tif'
    test_2_c(imageName)

    plt.show()
    dummy = 0
    
