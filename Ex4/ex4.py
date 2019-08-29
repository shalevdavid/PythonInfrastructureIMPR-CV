import numpy as np

def Fourier1D (x_n):

    N = np.size(x_n)

    x = np.arange(0, N, 1)
    k = np.arange(0, N, 1)

    X,K = np.meshgrid(x,k)

    Base = np.e**(-2*np.pi*K*X*1j/N)

    F_x_n = np.array(np.matrix(x_n) * np.matrix(Base).transpose())

    return F_x_n

def invFourier1D (F_n):

    N = np.size(F_n)

    x = np.arange(0, N, 1)
    k = np.arange(0, N, 1)

    K,X = np.meshgrid(k,x)

    Base = np.e**(2*np.pi*K*X*1j/N)

    x_n = np.array((np.matrix(F_n) * np.matrix(Base).transpose())/N)

    return x_n


def Fourier1DPolar (x_n):

    F_n = Fourier1D (x_n)
    theta = np.arctan2(np.imag(F_n), np.real(F_n))
    R = np.sqrt(F_n.real**2 + F_n.imag**2)

    F_n_polar = np.column_stack([R.transpose(), theta.transpose()])

    return F_n_polar


def  invFourier1DPolar (F_n_polar):

    R = F_n_polar[:, 0]
    theta = F_n_polar[:, 1]

    F_n = R*np.e**(1j*theta)

    x_n_rev = invFourier1D (F_n)

    return x_n_rev


def imageUpsampling (img, upsamplingFactor):

    scaleRow = upsamplingFactor[0]
    scaleCol = upsamplingFactor[1]

    Fimg = np.fft.fftshift(np.fft.fft2(img))

    if ( (scaleRow<1) or (scaleCol<1) ):
        return img,Fimg,Fimg

    r,c = Fimg.shape
    r2 = r*scaleRow
    c2 = c*scaleCol

    zeroPaddedFimg = np.zeros([r2,c2]).astype(np.complex)

    rowIndexStart = np.uint16(r2/2 - r/2)
    cowIndexStart = np.uint16(c2/2 - c/2)

    Fimg = Fimg * scaleRow * scaleCol

    zeroPaddedFimg[rowIndexStart:rowIndexStart+r, cowIndexStart:cowIndexStart+c] = Fimg

    #upsampledImage = (np.fft.ifft2(np.fft.ifftshift(zeroPaddedFimg)).real + np.fft.ifft2(np.fft.ifftshift(zeroPaddedFimg)).imag)*scaleRow*scaleCol
    #upsampledImage = (np.fft.ifft2(zeroPaddedFimg).real + np.fft.ifft2(zeroPaddedFimg).imag) * scaleRow * scaleCol
    upsampledImage = np.abs(np.fft.ifft2(np.fft.ifftshift(zeroPaddedFimg)))

    return upsampledImage, Fimg, zeroPaddedFimg


def phaseCorr (img1, img2):

    Nx,Ny = img1.shape

    ga = img1
    gb = img2

    Ga = np.fft.fft2(ga)
    Gb = np.fft.fft2(gb)

    R = Ga * np.conj(Gb) / np.abs(Ga * np.conj(Gb))
    r = np.fft.ifft2(R)
    #r = r/(r+r[-1:Nx*Ny-1])
    #r = r*np.conj(r)
    r = r.real

    maxIndFlat = np.argmax(r)
    rInd = np.uint16(maxIndFlat / Ny)
    cInd = np.uint16(maxIndFlat - rInd * Ny)

    res_dx = cInd
    res_dy = rInd
    phaseCorr = r

    return res_dx, res_dy, phaseCorr


def imFreqFilter(img, lowThresh, highThresh):

    Fimg = np.fft.fftshift(np.fft.fft2(img))

    Ny, Nx = img.shape
    a = Nx/2
    b = Ny/2

    x = np.arange(-a, Nx - a, 1)
    y = np.arange(-b, Ny - b, 1)
    X, Y = np.meshgrid(x, y)

    # indices = ((X-a)*(X-a) + (Y-b)*(Y-b) < r*r)
    #indices = (X * X + Y * Y < highThresh * highThresh)

    indices = np.logical_and( X * X + Y * Y >= lowThresh * lowThresh , X * X + Y * Y <= highThresh * highThresh )
    H = np.zeros_like(img)
    H[indices] = 1

    F_filtImage = H * Fimg
    filtImageTemp = np.fft.ifft2(F_filtImage)
    filtImage = np.abs(filtImageTemp)

    mask = H

    #maskTemp = np.fft.ifftshift(np.fft.ifft2(H))
    #mask = maskTemp.real + maskTemp.imag

    return filtImage, Fimg, mask

'''
def imageDeconv(img, kernel_degrad, k):

    Gimg = np.fft.fft2(img)

    kernel_degrad_extended = np.zeros(img.shape)
    r, c = kernel_degrad.shape
    kernel_degrad_extended[0:r, 0:c] = kernel_degrad
    shiftR = np.uint16(r/2)
    shiftC = np.uint16(c/2)
    kernel_degrad_extended = np.roll(kernel_degrad_extended, -shiftR, axis = 0)
    kernel_degrad_extended = np.roll(kernel_degrad_extended, -shiftC, axis = 1)

    H = np.fft.fft2(kernel_degrad_extended)
    Nu, Nv = Gimg.shape
    u = np.arange(0, Nu, 1)
    v = np.arange(0, Nv, 1)
    V, U = np.meshgrid(v, u)

    # FrecImg = np.conj(H) * Gimg / (H * np.conj(H) + k)
    FrecImg = np.conj(H) * Gimg / (H * np.conj(H) + k * (U ** 2 + V ** 2))
    recImgComplex = np.fft.ifft2(FrecImg)
    recImg = np.abs(recImgComplex)

    return recImg
'''

def imageDeconv(img, kernel_degrad, k):

    Gimg = np.fft.fftshift(np.fft.fft2(img))

    kernel_degrad_extended = np.zeros(img.shape)
    r, c = kernel_degrad.shape
    kernel_degrad_extended[0:r, 0:c] = kernel_degrad
    shiftR = np.uint16(r/2)
    shiftC = np.uint16(c/2)
    kernel_degrad_extended = np.roll(kernel_degrad_extended, -shiftR, axis = 0)
    kernel_degrad_extended = np.roll(kernel_degrad_extended, -shiftC, axis = 1)

    H = np.fft.fftshift(np.fft.fft2(kernel_degrad_extended))
    Nu, Nv = Gimg.shape
    u = np.arange(-Nu/2, Nu/2, 1)
    v = np.arange(-Nv/2, Nv/2, 1)
    V, U = np.meshgrid(v, u)

    # FrecImg = np.conj(H) * Gimg / (H * np.conj(H) + k)
    FrecImg = np.conj(H) * Gimg / (H * np.conj(H) + k * (U ** 2 + V ** 2))
    recImgComplex = np.fft.ifft2(np.fft.ifftshift(FrecImg))
    recImg = np.abs(recImgComplex)

    return recImg
