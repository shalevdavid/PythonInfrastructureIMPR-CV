import numpy as np


def getSampledImageAtResolution (dim, PixelSize, K=2):

    x = np.linspace(dim[0], dim[1], 1 / PixelSize + 1);
    y = np.linspace(dim[2], dim[3], 1 / PixelSize + 1);

    X,Y = np.meshgrid(x,y)

    I = np.cos(K * np.pi * (3 * X + 2 * Y));

    return I


def getImageHistogram(img):
    bars = np.arange(0, 257, 1)
    hist, bins = np.histogram(img, bars)
    return hist


def optimalQuantizationImage(img,K):

    # pre processing for simple calculations
    hist, bins = np.histogram(img, 256)
    histNorm = hist/np.size(img)
    accHistNorm = np.cumsum(histNorm)
    accCenterHistNorm = np.cumsum(np.linspace(0,255,256)*histNorm)

    numOfIteration = 100
    Z = np.linspace(0, 255, K + 1)  # initial guess
    #Z = np.ceil(np.random.rand(K+1)*K)    # initial guess
    #Z = np.flip((255 - np.ceil(255 * (np.log(np.arange(1, K + 2, 1)) / np.log(K + 2)))), axis=0)
    Q = np.zeros(K);

    for i in range(0,numOfIteration):
        for j in range(0,K):
            Zj1 = int(np.floor(Z[j]))
            Zj2 = int(np.floor(Z[j+1]))
            Q[j] = (accCenterHistNorm[Zj2] - accCenterHistNorm[Zj1])/(accHistNorm[Zj2] - accHistNorm[Zj1])
        for j in range(1,K):
            Z[j] = (Q[j]+Q[j-1])/2

    Z[0] = 0
    Z[np.size(Z)-1] = Z[np.size(Z)-1] + 1

    r = Q

    qImg = np.zeros(img.shape).astype(np.uint8)
    for i in range(0, np.size(r)):
        qImg[ np.where ( np.logical_and( img>=Z[i], img<Z[i+1] ) ) ] = np.round(r[i])

    #qImg = int(256/K)*(np.digitize(img, Z)-1).astype(np.uint8) # Shalev T.B.D
    return qImg


def getConstrastStrechedImage(img):

    imgMin = np.min(img)
    imgMax = np.max(img)
    imgLinearCostrast = (img - imgMin)/(imgMax-imgMin) # Shalev T.B.D - multiplication is not good enough
    imgLinearCostrast = 255*imgLinearCostrast

    return imgLinearCostrast.astype(np.uint8)

def getHistEqImage(img):

    alpha = 255/np.size(img)

    histA, bars = np.histogram(img,256)
    acchistA = np.cumsum(histA)

    imgEqRes = np.zeros_like(img);

    for x in range(0,np.size(img,axis=0)):
        for y in range(0,np.size(img,axis=1)):
            #print(x)
            #print(y)
            imgEqRes[x,y] = alpha*acchistA[img[x,y]];

    histEqRes = imgEqRes.astype(np.uint8);  # Shalev T.B.D
    return histEqRes