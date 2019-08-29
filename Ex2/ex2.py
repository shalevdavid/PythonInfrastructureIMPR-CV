import numpy as np

def getAffineTransformation(pts1,pts2):

    b = pts2.flatten()
    M = np.zeros([b.size,6])

    for j in np.arange(0,int(b.size/2),1):
        M[2*j] = [pts1[j,0], pts1[j,1], 0, 0, 1, 0]
        M[2*j+1] = [0, 0, pts1[j,0], pts1[j,1], 0, 1]

    p = np.linalg.lstsq(M,b)
    p = p[0]

    T = np.zeros([3, 3])

    [a, b, c, d, tx, ty] = p

    T = np.array([[a, b, tx],[c, d, ty],[0, 0, 1]])

    affineT = T
    return affineT

def applyAffineTransToImage(img, affineT):

  ### stage 1 - Perform Geometrix Mapping Transformation

    # [X,Y,W] = T*[xs,ys,1]

    Tmatrix = np.matrix(affineT)
    TmatrixInverseMapping = np.linalg.inv(Tmatrix)

    r,c = img.shape
    X,Y = np.meshgrid(np.linspace(0,c-1,c), np.linspace(0,r-1,r))
    destCoords = np.column_stack((X.flatten(), Y.flatten(), np.ones(r*c))).transpose() # destination Coords x,y

    sourceCoords = TmatrixInverseMapping*destCoords

  ### stage 2 - Pereform Bilinear interpolation

    imgT = biliinearInterploation(img,sourceCoords)
    return imgT

def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, p, b):

    Pd = Pt # dest = terget (for semantics only)
    Qd = Qt # dest = terget (for semantics only)

    # number of matching lines between source to destination images (k is num of multiple segments)
    K = np.size(Ps, axis=0)

    r, c = img.shape
    X, Y = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    xVec = X.flatten()
    yVec = Y.flatten()
    destCoords = np.column_stack((xVec,yVec))

    Rdest = destCoords

    RsourceK_MultiSegment = np.zeros([K, (r * c)]) # init to zeros (allocate size beforehand)
    WkRk_multiSeg = np.zeros([(r * c),2])
    Wk_multiSeg = np.zeros([(r * c),1])

    for k in range(0,K):
        Pdk = Pd[k]
        Qdk = Qd[k]
        PdkVec = np.tile(Pdk, (xVec.size, 1))

        udk = (Qdk - Pdk)/np.linalg.norm(Qdk-Pdk)
        vdk = [udk[1], -udk[0]]

        alpha = (np.matrix((Rdest - PdkVec))*np.matrix(udk).transpose())/np.linalg.norm(Qdk-Pdk)
        beta = np.matrix((Rdest - PdkVec)) * np.matrix(vdk).transpose()

        # Perform Inverse Mapping for segment k (one segment mapping)
        Psk = Ps[k]
        Qsk = Qs[k]
        PskVec = np.tile(Psk, (xVec.size, 1))
        usk = (Qsk - Psk) / np.linalg.norm(Qsk - Psk)
        vsk = [usk[1], -usk[0]]
        RsourceK = PskVec + alpha*np.linalg.norm(Qsk-Psk)*usk + beta*vsk
        RsourceK = np.array(RsourceK)

        a = 1e-10
        Wk = np.array((np.linalg.norm(Qsk-Psk)**p)/(a + np.abs(beta)))**b

        WkRk_multiSeg = WkRk_multiSeg + Wk*RsourceK
        Wk_multiSeg = Wk_multiSeg + Wk

    Rsource = WkRk_multiSeg / Wk_multiSeg

    # Perform Bilinear Interpolation
    sourceCoords = Rsource.transpose()
    imgT = biliinearInterploation(img, sourceCoords)

    return imgT


def biliinearInterploation(img, sourceCoords):

    r,c = img.shape

    xs = sourceCoords[0,:]
    ys = sourceCoords[1,:]

    xs = np.array(xs)
    ys = np.array(ys)

    xsLeft = np.uint16(np.ceil(xs)-1)
    xsRight = np.uint16(np.ceil(xs))
    ysDown = np.uint16(np.ceil(ys)-1)
    ysUp = np.uint16(np.ceil(ys))

    u = xs - xsLeft
    v = ys - ysDown

    indicesOutOfBounds1 = np.where(np.logical_or( xsLeft < 0, xsLeft >= c))
    indicesOutOfBounds2 = np.where(np.logical_or( xsRight < 0, xsRight >= c))
    indicesOutOfBounds3 = np.where(np.logical_or( ysDown < 0, ysDown >= r))
    indicesOutOfBounds4 = np.where(np.logical_or( ysUp < 0, ysUp >= r))

    xsRight[indicesOutOfBounds1] = 0
    xsRight[indicesOutOfBounds2] = 0
    xsRight[indicesOutOfBounds3] = 0
    xsRight[indicesOutOfBounds4] = 0
    xsLeft[indicesOutOfBounds1] = 0
    xsLeft[indicesOutOfBounds2] = 0
    xsLeft[indicesOutOfBounds3] = 0
    xsLeft[indicesOutOfBounds4] = 0
    ysDown[indicesOutOfBounds1] = 0
    ysDown[indicesOutOfBounds2] = 0
    ysDown[indicesOutOfBounds3] = 0
    ysDown[indicesOutOfBounds4] = 0
    ysUp[indicesOutOfBounds1] = 0
    ysUp[indicesOutOfBounds2] = 0
    ysUp[indicesOutOfBounds3] = 0
    ysUp[indicesOutOfBounds4] = 0
    xs[indicesOutOfBounds1] = 0
    xs[indicesOutOfBounds2] = 0
    xs[indicesOutOfBounds3] = 0
    xs[indicesOutOfBounds4] = 0
    ys[indicesOutOfBounds1] = 0
    ys[indicesOutOfBounds2] = 0
    ys[indicesOutOfBounds3] = 0
    ys[indicesOutOfBounds4] = 0

    imgVec = np.array(np.reshape(img, (r * c, 1))).flatten()
    SE = imgVec[ysDown*c + xsRight]
    SW = imgVec[ysDown * c + xsLeft]
    NE = imgVec[ysUp * c + xsRight]
    NW = imgVec[ysUp * c + xsLeft]

    Su = SE*u + SW*(1-u)
    Nu = NE*u + NW*(1-u)
    Vuv = Nu*v + Su*(1-v)

    Vuv[indicesOutOfBounds1] = 0
    Vuv[indicesOutOfBounds2] = 0
    Vuv[indicesOutOfBounds3] = 0
    Vuv[indicesOutOfBounds4] = 0
    newImage = np.reshape(Vuv, (r,c))

    imgT = newImage
    return imgT


def imGradSobel(img):

    #for Sx = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
    Sxx = np.array([-1, 0, 1])
    Sxy = np.array([1, 2, 1])

    '''
    #for Sx = np.array([[1, 0, -1], [2, 0, -2],[1, 0, -1]])
    Sxx = np.array([1, 0, -1])
    Sxy = np.array([1, 2, 1])
    '''

    Sxx = np.array(Sxx)[None, :]  # Sxx.shape = (1, 3), instead of (3,)
    Sxy = np.array(Sxy)[:, None]  # Sxx.shape = (3, 1), instead of (3,)

    # Gx = img*Sx = img*Sxx*Sxy
    Gx = convolution2D(convolution2D(img,Sxx), Sxy)

    # Gy = img*Sy = img*Syx*Syy
    Syx = Sxx.transpose()
    Syy = Sxy.transpose()
    Gy = convolution2D(convolution2D(img, Syx), Syy)

    # Gmag = gradiant = sqrt(Gx^2+Gy^2)
    Gmag = np.sqrt(Gx**2 + Gy**2)

    return Gx,Gy,Gmag

'''
def convolution2D(img, h):

    from scipy import signal
    import matplotlib.pyplot as plt

    convRes = signal.convolve2d(img, h)

    return convRes

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
    imgExtended[rStart:rStart+r, cStart:cStart+c] = img

    ## Extending the image using mirroring approach
    if (rk > 1):
        imgExtended[0:rStart, cStart:cStart + c] = img[np.flip(np.arange(0, rStart), axis=0), :]
        imgExtended[r+rStart:r+rStart+rStart, cStart:cStart + c] = img[np.flip(np.arange(r-rStart-1,r-1), axis=0), :]
    if (ck > 1):
        imgExtended[:, 0:cStart] = imgExtended[:, np.flip(np.arange(cStart, cStart + cStart), axis=0)]
        imgExtended[:, c+cStart:c+cStart+cStart] = imgExtended[:, np.flip(np.arange(c-1, c+cStart-1), axis=0)]

    Kernel = np.flip(np.flip(h, axis=0), axis=1)

    # convolution result "same"
    convRes = np.zeros([r, c])
    for x in range(0, c):
        for y in range(0, r):
            convRes[y, x] = np.sum(Kernel * imgExtended[y:y+rk, x:x+ck])

    return convRes

'''
# another implementation: convRes here is using the "full" approach, imgRes result is using the "full" approch
# and extension approach done by continue of the origin image
def convolution2D(img, h):

    r, c = img.shape
    rk, ck = h.shape

    convRes = np.zeros([r + 2 * rk - 2, c + 2 * ck - 2])
    imgExtended = np.zeros([r + 2 * rk - 2, c + 2 * ck - 2])
    imgExtended[(rk - 1):(r + rk - 1),
    (ck - 1):(c + ck - 1)] = img  ## convolution is Space invariant, using this fact later

    ## Extending the image
    if (rk > 1):
        imgExtended[0:(rk - 1), :] = np.tile(imgExtended[rk - 1, :], [rk - 1, 1])
        imgExtended[(r + rk - 1):(r + 2 * rk - 2), :] = np.tile(imgExtended[r + rk - 2, :], [rk - 1, 1])
    if (ck > 1):
        imgExtended[:, 0:(ck - 1)] = np.tile(imgExtended[:, ck - 1], [ck - 1, 1]).transpose()
        imgExtended[:, (c + ck - 1):(c + 2 * ck - 2)] = np.tile(imgExtended[:, c + ck - 2], [ck - 1, 1]).transpose()

    Kernel = np.flip(np.flip(h, axis=0), axis=1)

    for x in range(0, c + 2 * ck - 2 - 2):
        for y in range(0, r + 2 * rk - 2 - 2):
            convRes[y, x] = np.sum(Kernel * imgExtended[y:(y + rk), x:(x + ck)])

    ## Convolution is Space Invariant + "same" approach result into imgRes
    imgRes = convRes[(rk - 1):(r + rk - 1), (ck - 1):(c + ck - 1)]

    return imgRes

'''
