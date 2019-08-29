import numpy as np
#import scipy.signal as scisig

'''
# using "Same" approach size + mirroring extension
def convolution2D(img, h):
    r, c = img.shape
    rk, ck = h.shape

    # for extension
    rExt = r + rk - 1
    cExt = c + ck - 1

    rStart = np.uint8(rk / 2)
    cStart = np.uint8(ck / 2)

    imgExtended = np.zeros([rExt, cExt])
    imgExtended[rStart:rStart + r, cStart:cStart + c] = img

    ## Extending the image using mirroring approach
    if (rk > 1):
        imgExtended[0:rStart, cStart:cStart + c] = img[np.flip(np.arange(0, rStart), axis=0), :]
        imgExtended[r + rStart:r + rStart + rStart, cStart:cStart + c] = img[np.flip(np.arange(r - rStart - 1, r - 1),
                                                                                     axis=0), :]
    if (ck > 1):
        imgExtended[:, 0:cStart] = imgExtended[:, np.flip(np.arange(cStart, cStart + cStart), axis=0)]
        imgExtended[:, c + cStart:c + cStart + cStart] = imgExtended[:,
                                                         np.flip(np.arange(c - 1, c + cStart - 1), axis=0)]

    Kernel = np.flip(np.flip(h, axis=0), axis=1)

    # convolution result "same"
    convRes = np.zeros([r, c])
    for x in range(0, c):
        for y in range(0, r):
            convRes[y, x] = np.sum(Kernel * imgExtended[y:y + rk, x:x + ck])

    return convRes
'''

def convolution2D(img, h):

    hExt = np.zeros(img.shape)
    r,c = h.shape
    hExt[0:r, 0:c] = h

    shiftR = np.uint16(r / 2)
    shiftC = np.uint16(c / 2)
    hExt = np.roll(hExt, -shiftR, axis=0)
    hExt = np.roll(hExt, -shiftC, axis=1)

    convRes = np.fft.ifft2(np.fft.fft2(img)*np.fft.fft2(hExt)).real

    return convRes


def getKernel(filterParam):

    if (filterParam <= 0.25):
        return -1 # error

    a = filterParam
    b = 0.25
    c = 0.25 - a/2

    kernel1D = np.array([c,b,a,b,c])

    return kernel1D

def imConv2(img, kernel1D):

    # return convolution2D(convolution2D(img,kernel1D[None, :]), kernel1D[:, None])

    rIm,cIm = img.shape
    kernel1D_Extended = np.zeros([1,cIm])
    r = np.size(kernel1D)
    kernel1D_Extended[0, 0:r] = kernel1D
    shiftR = np.uint16(r / 2)
    kernel1D_Extended = np.roll(kernel1D_Extended, -shiftR)

    FFT_Kernel1D = np.fft.fft(kernel1D_Extended)

    # FFT_Kernel2D = FFT_Kernel1D*FFT_Kernel1D.transpose()
    # convRes = np.fft.ifft2(np.fft.fft2(img) * FFT_Kernel2D).real
    # return convRes

    return np.fft.ifft2(np.fft.fft2(img) * FFT_Kernel1D*FFT_Kernel1D.transpose()).real

def gaussianPyramid (img, numOfLevels, filterParam):

    if (filterParam <= 0.25):
        return -1 # error

    a = filterParam
    b = 0.25
    c = 0.25 - a/2

    g = np.array([c,b,a,b,c])
    LPFgaussian = np.array(np.matrix(g).transpose()*np.matrix(g))

    g_pyr = {0 : img}
    imgPrev = img
    for i in range(1,numOfLevels):
        # Gaussian LPF
        #imgCurLevel = scisig.convolve2d(imgPrev, LPFgaussian, 'same')
        imgCurLevel = convolution2D(imgPrev, LPFgaussian)
        #imgCurLevel = imConv2(imgPrev, getKernel(filterParam))

        # DownSampling by factor of 2
        r,c = imgPrev.shape
        rowIndices = np.arange(0, r, 2)
        colIndices = np.arange(0, c, 2)
        imgCurLevel = imgCurLevel[rowIndices, :]
        imgCurLevel = imgCurLevel[:, colIndices]

        # inserting to dictionary and preparing next level generation
        g_pyr[i] = imgCurLevel
        imgPrev = imgCurLevel

    return  g_pyr


def laplacianPyramid (img, numOfLevels, filterParam):

    g_pyr = gaussianPyramid(img, numOfLevels, filterParam)

    l_pyr = { numOfLevels-1 : g_pyr[numOfLevels-1]}

    # run on gaussian pyramid from high level to low level, in order to construct laplacian pyramid
    for i in np.arange(numOfLevels-2,-1,-1): # i running from numOfLevels-2 downto 0 (including), theses are the numOfLevels-1 levels left
        # expanding
        expanded = ExpandOperation(g_pyr[i+1], filterParam) # expending l_pyr_recon
        # save lower level
        l_pyr[i] = g_pyr[i] - expanded

    return l_pyr

def imgFromLaplacianPyramid(l_pyr, numOfLevels, filterParam):

    l_pyr_recon = { numOfLevels-1 : l_pyr[numOfLevels-1]}

    for i in np.arange(numOfLevels-2,-1,-1): # i running from numOfLevels-2 downto 0 (including), theses are the numOfLevels-1 levels left
        # expanding
        l_pyr_recon_expanded = ExpandOperation(l_pyr_recon[i+1], filterParam) # expending l_pyr_recon
        # reconstruct current level
        l_pyr_recon[i] = l_pyr_recon_expanded + l_pyr[i]

    return l_pyr_recon[0].round()


# ExpandOperation = imageUpsampling
def ExpandOperation(img, filterParam):

    if (filterParam <= 0.25):
        return -1 # error

    a = filterParam
    b = 0.25
    c = 0.25 - a/2

    g = np.array([c,b,a,b,c])
    LPFgaussian = np.array(np.matrix(g).transpose()*np.matrix(g))

    scaleRow = 2
    scaleCol = 2
    r,c = img.shape
    r2 = r*scaleRow
    c2 = c*scaleCol

    extendedImg = np.zeros((r2, c2))
    extendedImg[::scaleRow, ::scaleCol] = img*scaleRow*scaleCol
    #extendedImgRes = scisig.convolve2d(extendedImg, LPFgaussian, 'same')
    extendedImgRes = convolution2D(extendedImg, LPFgaussian)
    #extendedImgRes = imConv2(extendedImg, getKernel(filterParam))

    return extendedImgRes


def imgBlending(img1, img2, blendingMask, numOfLevels, filterParam):

    l_pyr_1 = laplacianPyramid(img1, numOfLevels, filterParam)
    l_pyr_2 = laplacianPyramid(img2, numOfLevels, filterParam)

    r,c = blendingMask.shape
    i = numOfLevels - 1
    rowIndices = np.arange(0, r, 2**i)
    colIndices = np.arange(0, c, 2**i)
    blendingMaskCurLevel = blendingMask
    blendingMaskCurLevel = blendingMaskCurLevel[rowIndices, :]
    blendingMaskCurLevel = blendingMaskCurLevel[:, colIndices]

    blendedImg = blendingMaskCurLevel*l_pyr_1[numOfLevels - 1] + (1-blendingMaskCurLevel)*l_pyr_2[numOfLevels - 1]

    for i in np.arange(numOfLevels - 2, -1, -1):
        expanded = ExpandOperation(blendedImg, filterParam)

        rowIndices = np.arange(0, r, 2 ** i)
        colIndices = np.arange(0, c, 2 ** i)
        blendingMaskCurLevel = blendingMask
        blendingMaskCurLevel = blendingMaskCurLevel[rowIndices, :]
        blendingMaskCurLevel = blendingMaskCurLevel[:, colIndices]

        blendedImg = expanded + blendingMaskCurLevel*l_pyr_1[i] + (1-blendingMaskCurLevel)*l_pyr_2[i]

    return blendedImg
